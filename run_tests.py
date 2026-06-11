"""Run the three-step test sequence non-interactively."""
from __future__ import annotations
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from lab_team.config import load_all
from lab_team.crew import ask, build_crew

TESTS = [
    "Create project Suzuki-Optimization, then log experiment EXP-12: 5 mL acetone (solvent) and 0.1 mmol palladium acetate (catalyst).",
    "Any chemicals expiring in 90 days or missing SDS?",
    "Write a Word report summarizing project Suzuki-Optimization with a yield bar chart.",
]

def main() -> int:
    load_all()

    for i, prompt in enumerate(TESTS, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {prompt}")
        print('='*60)
        # Rebuild crew for each test: CrewAI mutates task.agent and the
        # manager's delegation tool roster during kickoff(), so reusing a
        # single crew instance across calls corrupts the roster on test 2+.
        print("Assembling the lab team…")
        crew = build_crew()
        print("Crew ready.")
        try:
            answer = ask(crew, prompt)
            print(f"\nManager: {answer}\n")
        except Exception as exc:
            print(f"\n[Error] {type(exc).__name__}: {exc}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
