"""CLI entry point: `python -m eln --demo`.

Imports `run` from the package (a single copy of the module, no double-import).
"""

from __future__ import annotations

import json
import sys

from eln.flow import run


def demo() -> None:
    demo_text = (
        "Ran a coupling with IntermediateX-119 in toluene at 110 C for 8 h, "
        "2 mol% catalyst, 3 bar, purity 97."
    )
    state = run(demo_text, question="which solvent gives the highest yield?")
    print("\n=== RECORD ===", json.dumps(state.record, indent=2))
    print("=== INVENTORY ===", state.inventory_result)
    print("=== PREDICTION ===", state.prediction)
    print("=== ERRORS ===", state.errors)
    print("=== REPORT ===\n", state.report)


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        print("usage: python -m eln --demo")
