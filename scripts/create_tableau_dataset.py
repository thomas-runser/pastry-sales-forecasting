from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tableau_dataset import write_tableau_exports


if __name__ == "__main__":
    fact_output, calendar_output = write_tableau_exports(PROJECT_ROOT)

    # Read the files that were actually written so the command also verifies
    # the final public exports and their paths.
    fact = pd.read_csv(fact_output, parse_dates=["Date"])
    calendar = pd.read_csv(calendar_output, parse_dates=["Date"])

    print("Created fact table:", fact_output)
    print("Fact rows:", len(fact))
    print("Fact columns:", len(fact.columns))
    print("Created calendar table:", calendar_output)
    print("Calendar rows:", len(calendar))
    print("Unique calendar dates:", calendar["Date"].nunique())
    print(
        "Date range:",
        calendar["Date"].min().date(),
        "to",
        calendar["Date"].max().date(),
    )
    print("Stores:", fact["Store"].nunique())
    print("Products:", fact["Product"].nunique())
