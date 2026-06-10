"""ChatEngine — hybrid conversational engine, the only API the UI knows.

Cloud path (key configured): CrewAI orchestrator + EHS agent with the
sanitised tool belt. Local path (no key, or cloud failure): deterministic
router + existing template writers. Both paths share one audited IP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import llm
from ..crew_safety import write_safety_briefing
from ..inventory import Inventory
from ..rag import RAG
from ..safety import check_safety
from ..sanitizer import Sanitizer, audit
from ..tools import parse_experiment, predict_outcome
from .router import classify_intent, extract_reagents

HISTORY_WINDOW = 12

_LOCAL_CAPABILITIES = (
    "I am running in local mode (no cloud LLM). I can answer questions about:\n"
    "- **safety** — e.g. \"Is it safe to mix bleach with ammonia?\"\n"
    "- **prediction** — e.g. \"What yield should I expect at 110C in toluene?\"\n"
    "- **history** — e.g. \"Did we run a similar coupling before?\"\n"
    "- **inventory** — e.g. \"How much toluene is left in stock?\""
)


@dataclass
class ChatReply:
    text: str
    critical: bool = False
    mode: str = "local"  # "cloud" | "local" | "local_fallback"


class ChatEngine:
    def __init__(self):
        self.sanitizer = Sanitizer()

    def reply(self, history: list[dict], message: str) -> ChatReply:
        """history: [{"role": "user"|"assistant", "content": str}, ...]"""
        reagents = extract_reagents(message)  # raw text, local, BEFORE scrub
        clean_msg, redacted = self.sanitizer.scrub(message)

        if llm.has_cloud():
            audit(operation="chat_turn", sent_to_cloud=clean_msg,
                  redacted=redacted, stayed_local=["raw message", "reagent names"])
            try:
                return self._cloud_reply(history, clean_msg, reagents)
            except Exception:
                local = self._local_reply(message, reagents)
                return ChatReply(
                    text=local.text + "\n\n_answered locally (cloud unavailable)_",
                    critical=local.critical, mode="local_fallback")

        audit(operation="chat_turn", sent_to_cloud="", redacted=redacted,
              stayed_local=["raw message", "reagent names", "full local answer"])
        return self._local_reply(message, reagents)

    def _sanitized_transcript(self, history: list[dict]) -> str:
        """Last HISTORY_WINDOW messages, scrubbed — history stores REAL names locally."""
        window = history[-HISTORY_WINDOW:]
        return "\n".join(
            f"{m['role']}: {self.sanitizer.scrub(m['content'])[0]}" for m in window)

    # ---- local path (deterministic) ----

    def _local_reply(self, message: str, reagents: list[str]) -> ChatReply:
        intent = classify_intent(message)
        if intent == "safety":
            if not reagents:
                return ChatReply(text="Which reagents do you plan to combine? "
                                      "Name them and I will run the safety check.")
            report = check_safety(reagents)
            return ChatReply(text=write_safety_briefing(report),
                             critical=report.severity == "critical")
        if intent == "predict":
            record = parse_experiment(message)
            pred = predict_outcome(record)
            if "error" in pred:
                return ChatReply(text=f"Prediction unavailable: {pred['error']}")
            return ChatReply(text=(
                f"Predicted yield: **{pred['predicted_yield_pct']}%** · "
                f"predicted QC pass: **{pred['predicted_qc_pass']}** "
                f"(solvent={record['solvent']}, T={record['temperature_c']}C, "
                f"t={record['time_h']}h)"))
        if intent == "history":
            context = RAG().context(message, k=4)
            return ChatReply(text="From the local knowledge base:\n\n" + context)
        if intent == "inventory":
            inv = Inventory()
            df = inv.df[inv.df["name"].isin(reagents)] if reagents else inv.df
            lines = [f"- {r['name']}: {r['quantity_g']} g · expires {r['expiry']} "
                     f"· {r['location']}" for _, r in df.head(10).iterrows()]
            return ChatReply(text="Current stock:\n" + "\n".join(lines))
        if intent == "log_experiment":
            record = parse_experiment(message)
            return ChatReply(text="Structured record (use the *Log experiment* tab "
                                  f"to run the full flow):\n```json\n{record}\n```")
        return ChatReply(text=_LOCAL_CAPABILITIES)

    # ---- cloud path (CrewAI orchestrator + EHS agent, sanitised tools) ----

    def _cloud_reply(self, history: list[dict], clean_msg: str,
                     reagents: list[str]) -> ChatReply:
        from crewai import Agent, Crew, Task

        from ..crew_safety import ehs_agent
        from .safe_tools import SAFE_TOOLS

        transcript = self._sanitized_transcript(history)
        aliases = [self.sanitizer.scrub(r)[0] for r in reagents]

        assistant = Agent(
            role="Lab Assistant",
            goal="Answer the chemist using the lab's tools; ground every claim "
                 "in tool results, never invent hazard data.",
            backstory="A meticulous R&D assistant. Reagent names like REAGENT_A1 "
                      "are anonymised aliases: use them verbatim, never guess "
                      "what they might be.",
            tools=SAFE_TOOLS,
            llm=llm.get_llm(),
            verbose=False,
        )
        task = Task(
            description=(
                f"Conversation so far:\n{transcript}\n\n"
                f"New message from the chemist:\n{clean_msg}\n\n"
                f"Reagents detected in the message (aliases): {aliases}\n"
                "Answer the new message. If the question involves combining "
                "reagents, you MUST call check_safety and report its findings "
                "faithfully, including severity."
            ),
            expected_output="A concise, factual answer in Markdown.",
            agent=assistant,
        )
        crew = Crew(agents=[assistant, ehs_agent()], tasks=[task], verbose=False)
        answer = str(crew.kickoff())

        critical = False
        if reagents:
            critical = check_safety(reagents).severity == "critical"
        return ChatReply(text=self.sanitizer.unscrub(answer),
                         critical=critical, mode="cloud")
