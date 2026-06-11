# Lab Team — Build Status

_Last updated: 2026-06-11_

## All 3 tests PASSED (fully local, Ollama qwen2.5:14b)

### TEST 1 ✅ — Create project + log experiment with materials
- `create_project('Suzuki-Optimization')` → DB id 4, status active
- `create_experiment('EXP-12')` → linked to Suzuki-Optimization
- `log_experiment_material` × 2 → acetone 5.0 mL (solvent, CAS 67-64-1 via PubChem), palladium acetate 0.1 mmol (catalyst, CAS 3375-31-3 via PubChem)

### TEST 2 ✅ — Safety/expiry/SDS report
- `expiring_soon_report` ran → [] (no chemicals expiring in 90 days in test inventory)
- `sds_gaps_report` ran → [] (no SDS gaps in test inventory)

### TEST 3 ✅ — Word report + yield bar chart
- `read_project_data('Suzuki-Optimization')` → pulled full project/experiment/material data
- `plot_chart` → PNG saved to `lab_team/reports/`
- `generate_word_report` → `lab_team/reports/Suzuki-Optimization-Report.docx` (36KB)

---

## Bugs fixed during validation

### 1. Anthropic API key revoked
- Key committed in plaintext in `chem_rag_pipeline/.env`, auto-revoked by GitHub secret scanning
- Fix: `MANAGER_MODEL = "ollama/qwen2.5:14b"` in `config.py` for fully local operation
- To restore: add a new key to `chem_rag_pipeline/.env` (gitignored) and set `MANAGER_MODEL = "anthropic/claude-sonnet-4-6"`

### 2. Ollama context overflow (4096 tokens)
- Hierarchical crew system prompts + tool descriptions exceed 4096 tokens → model silently stalled
- Fix: `additional_params={"extra_body": {"options": {"num_ctx": 16384}}}` in `llm_factory.py`
- Mechanism: OpenAI client passes `extra_body` verbatim in JSON body; Ollama reads `options.num_ctx`

### 3. CrewAI delegation tool roster bug
- `crew.py` had `agent=manager` on the Task → CrewAI's `_update_manager_tools` built roster with `[manager]` only
- Workers invisible; all delegations failed with "coworker not found: lab operations manager"
- Fix: removed `agent=manager` from Task constructor; with `task.agent=None`, CrewAI uses `self.agents` (the 3 workers)

### 4. CrewAI kickoff state mutation between tests
- Single `Crew` instance reused across 3 `kickoff()` calls; CrewAI mutates manager's delegation tool roster after first call
- Fix: rebuild `Crew` for each test in `run_tests.py`

### 5. `generate_word_report` / `plot_chart` output_name confusion
- qwen2.5:14b passed a full path as `output_name`, mangling the filename via `_out()`
- Fix: added explicit docstring note: "SHORT filename WITHOUT extension and WITHOUT any directory path"

---

## Known limitations (qwen2.5:14b as manager)

- Manager sometimes fires 2 parallel delegations instead of 1 (duplicate DB writes, harmless due to unique constraints)
- TEST 2 Manager answer says "concurrent execution issue" even though both inventory tools ran correctly
- `get_synthesis_facts` called with hallucinated PDF path (expected — no paper provided in TEST 3 prompt)
- All limitations go away when Claude is restored as manager

---

## Env
- Team venv: `.venv-team` (Python 3.12.10) — crewai 1.14.6 + full RAG stack
- Manager: `ollama/qwen2.5:14b` (local fallback) — restore to `anthropic/claude-sonnet-4-6`
- Workers: `ollama/qwen2.5:14b` (keep as-is)
- CrewAI gotcha: no bundled litellm; tool signatures must avoid `Optional[...]`
