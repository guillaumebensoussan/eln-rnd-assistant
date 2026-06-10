# ELN Chat Assistant — Design

**Date:** 2026-06-10
**Status:** Approved by Guillaume (sections validated one by one)
**Builds on:** SDS safety & compatibility module (merged 2026-06-10), `ehs_agent()` seam in `src/eln/crew_safety.py`

## Goal

A conversational assistant tab in the Streamlit app where a chemist types free text
("Tomorrow I'll couple IntermediateX-119 with bleach in toluene — safe?") and gets
answers drawing on **all** ELN capabilities: SDS safety check, yield/QC prediction,
RAG over local history, and inventory. Multi-turn, with session memory.

Two non-negotiable constraints:

1. **IP-safe**: proprietary reagent names never reach a cloud LLM — in user
   messages, in tool inputs, or in tool outputs. Every cloud-bound byte passes
   through the existing `Sanitizer`, and every turn is recorded in the IP audit log.
2. **Graceful degradation**: without an API key the chat still works, via a
   deterministic keyword router and the existing template writers. No dead tab.

## Architecture (chosen: hybrid)

Two reply paths behind one engine, sharing a single audited IP boundary:

- **Cloud path** (key present): a CrewAI orchestration — a "Lab Assistant"
  conversational agent plus the existing `ehs_agent()` (this is the seam that
  module was built for) — equipped with a **sanitised tool belt**.
- **Local path** (no key, or cloud call fails): deterministic intent router +
  existing template writers (`write_safety_briefing`, raw prediction, RAG excerpts).

Rejected alternatives: pure deterministic routing everywhere (poor free-text
understanding, no agent in the loop for an agentic-AI final project) and pure
CrewAI (no degraded mode, which Guillaume explicitly wants).

## Components

### New: `src/eln/chat/router.py`
Pure, offline, fully testable.
- `classify_intent(text) -> Intent` — keyword/regex classification into
  `safety | predict | history | inventory | log_experiment | unknown`.
- `extract_reagents(text) -> list[str]` — match reagent names against the full
  inventory (longest-name-first matching, same technique as `Sanitizer`, but over
  **all** names, not only `ip_sensitive` ones). Runs on **raw** text, locally —
  after sanitisation the aliases would no longer match.

### New: `src/eln/chat/safe_tools.py`
CrewAI tool wrappers around the existing tools (`check_safety`, `search_history`,
`predict_outcome`, `parse_experiment`) enforcing the IP boundary structurally:
- **Inbound**: the agent only ever sees aliases; wrappers translate alias → real
  name locally (the `Sanitizer` alias map is bijective and stable) before
  executing the underlying tool.
- **Outbound**: every tool result is passed through `Sanitizer.scrub()` before
  being returned to the agent.
- Each tool call writes an audit entry.

### New: `src/eln/chat/engine.py`
`ChatEngine.reply(history, message) -> ChatReply`. The only API the UI knows.
1. Extract reagents locally (router) from raw text; keep real names in local state.
2. `Sanitizer.scrub()` the message; write a `chat_turn` audit entry
   (redacted count, chars sent to cloud — 0 on the local path).
3. Cloud path: orchestrator agent + EHS agent with `safe_tools`, fed the sanitised
   message and a sanitised window of the last 12 messages.
4. Local path: intent → direct call of local functions → template answer.
5. **De-sanitise the final answer locally** (alias → real name, inverse mapping)
   so the chemist reads "IntermediateX-119", never "REAGENT_A3".
6. If a safety result is `critical`, the reply carries a flag so the UI can show
   the ⛔ banner and require acknowledgement.

### Modified: `src/eln/llm.py`
- `ELN_MODEL` env var, default `anthropic/claude-sonnet-4-5`. Any LiteLLM
  provider string works (documented examples: `gemini/gemini-2.5-flash`,
  `openai/gpt-4o-mini`).
- `has_cloud()` checks for the key matching the configured provider prefix
  (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`).
- `.env.example` documents the three cases. Whole app benefits (everything goes
  through `get_llm()`).

### Modified: `app/streamlit_app.py`
- New "Assistant" tab, first position: `st.chat_input`, history in
  `st.session_state`, mode banner (cloud provider name / "local mode").
- Critical safety replies render the ⛔ banner and require the same
  acknowledgement checkbox as the Safety tab (same `safety_acknowledgement`
  audit entry) before the conversation continues.

## Data flow — IP boundary (the load-bearing decisions)

- Reagent extraction happens **before** sanitisation, on raw text, locally.
- Real names live only in local state; the cloud LLM sees aliases everywhere —
  user messages, history window, tool inputs, tool outputs.
- The displayed answer is de-sanitised locally so UX is unaffected.
- The chat shows up in the IP Audit log tab like every other operation
  (strong demo argument).

## Error handling (all non-blocking, like the rest of the app)

- **Cloud call fails mid-chat** (timeout, quota, bad key): fall back to the local
  path for that turn, with a discreet note "answered locally (cloud unavailable)".
- **Unknown intent on the local path**: honest reply listing what local mode can
  do (safety / prediction / history / inventory) with an example phrasing. No
  capability hallucination.
- **Reagent not in inventory**: explicit "not in inventory — no safety data",
  consistent with the SDS module's `unknowns`.

## Testing (offline, same standard as the SDS module)

- `router.py`: intents; reagent extraction (multi-word names, case,
  `IntermediateX-###`).
- `safe_tools.py`: **the test that matters** — no `ip_sensitive` name survives in
  any cloud-bound payload; alias → real → alias round-trip is bijective.
- `engine.py`: end-to-end local path with `has_cloud` monkeypatched to False:
  incompatible reagents → critical briefing; history question → RAG excerpts;
  display de-sanitisation.
- `llm.py`: `ELN_MODEL` → correct key required.
- No cloud-path test in CI (no key) — same policy as the existing code.

## Out of scope (YAGNI)

Conversation persistence across sessions, response streaming, vector memory of
the chat, multi-user support.
