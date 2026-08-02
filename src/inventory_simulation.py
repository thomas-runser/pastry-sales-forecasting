"""FIFO inventory simulation for two-day pastry shelf life.

The forecast is interpreted as the desired total stock available for the sales
Date. Existing carryover stock is sold first. Only new production left at the
end of the day can carry to the next calendar day. Unsold carryover stock
expires at the end of its second sales day and is counted as simulated loss.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


REQUIRED_KEYS = ["Date", "Store", "Product"]


def _nonnegative_number(value: object, default: float = 0.0) -> float:
    """Convert a value to a finite nonnegative float."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return float(default)
    return float(max(numeric, 0.0))


def simulate_fifo_strategy(
    evaluation: pd.DataFrame,
    forecast_column: str,
    *,
    demand_column: str = "Demand",
    initial_carryover_column: str = "CarryoverStock",
    model_name: str | None = None,
) -> pd.DataFrame:
    """Simulate forecast-driven production with FIFO and a two-day shelf life.

    Production for each day is calculated as::

        max(ForecastDemand - OpeningCarryover, 0)

    Opening carryover is sold before newly produced stock. Any opening
    carryover still unsold at day-end becomes ``SimulatedLoss``. Newly produced
    stock still unsold becomes ``ClosingCarryover`` and can be sold on the next
    calendar day.

    A missing calendar day breaks the simulated sequence because demand for the
    missing day is unknown. At each new sequence, the actual
    ``initial_carryover_column`` value is used when available; otherwise zero is
    used.
    """
    required_columns = {
        *REQUIRED_KEYS,
        demand_column,
        forecast_column,
    }
    missing_columns = required_columns.difference(evaluation.columns)
    if missing_columns:
        raise KeyError(
            "Missing required simulation columns: "
            + str(sorted(missing_columns))
        )

    data = evaluation.copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.normalize()
    data[demand_column] = pd.to_numeric(data[demand_column], errors="coerce")
    data[forecast_column] = pd.to_numeric(
        data[forecast_column],
        errors="coerce",
    )

    data = data.loc[
        data[demand_column].notna()
        & data[forecast_column].notna()
        & data[demand_column].ge(0)
    ].copy()

    data = data.sort_values(REQUIRED_KEYS).reset_index(drop=True)

    rows: list[dict[str, object]] = []

    for (_, _), group in data.groupby(
        ["Store", "Product"],
        sort=False,
        dropna=False,
    ):
        group = group.sort_values("Date")
        simulated_carryover = 0.0
        previous_date: pd.Timestamp | None = None

        for _, source_row in group.iterrows():
            date = pd.Timestamp(source_row["Date"]).normalize()
            sequence_start = (
                previous_date is None
                or date != previous_date + pd.Timedelta(days=1)
            )

            if sequence_start:
                initial_value = (
                    source_row.get(initial_carryover_column, 0.0)
                    if initial_carryover_column in source_row.index
                    else 0.0
                )
                opening_carryover = _nonnegative_number(initial_value)
                initial_carryover_added = opening_carryover
            else:
                opening_carryover = simulated_carryover
                initial_carryover_added = 0.0

            actual_demand = _nonnegative_number(source_row[demand_column])
            forecast_demand = round(
                _nonnegative_number(source_row[forecast_column])
            )

            recommended_production = max(
                forecast_demand - opening_carryover,
                0.0,
            )
            available_stock = opening_carryover + recommended_production

            old_stock_sold = min(actual_demand, opening_carryover)
            demand_after_old_stock = max(
                actual_demand - old_stock_sold,
                0.0,
            )
            new_stock_sold = min(
                demand_after_old_stock,
                recommended_production,
            )

            fulfilled_sales = old_stock_sold + new_stock_sold
            stockout_units = max(actual_demand - fulfilled_sales, 0.0)
            simulated_loss = max(
                opening_carryover - old_stock_sold,
                0.0,
            )
            closing_carryover = max(
                recommended_production - new_stock_sold,
                0.0,
            )

            row = source_row.to_dict()
            row.update(
                {
                    "Model": model_name or forecast_column,
                    "ForecastDemand": forecast_demand,
                    "ActualDemand": actual_demand,
                    "SequenceStart": sequence_start,
                    "InitialCarryoverAdded": initial_carryover_added,
                    "OpeningCarryover": opening_carryover,
                    "RecommendedProduction": recommended_production,
                    "AvailableStock": available_stock,
                    "OldStockSold": old_stock_sold,
                    "NewStockSold": new_stock_sold,
                    "FulfilledSales": fulfilled_sales,
                    "StockoutUnits": stockout_units,
                    "SimulatedLoss": simulated_loss,
                    "ClosingCarryover": closing_carryover,
                }
            )
            rows.append(row)

            simulated_carryover = closing_carryover
            previous_date = date

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    numeric_output_columns = [
        "ForecastDemand",
        "ActualDemand",
        "InitialCarryoverAdded",
        "OpeningCarryover",
        "RecommendedProduction",
        "AvailableStock",
        "OldStockSold",
        "NewStockSold",
        "FulfilledSales",
        "StockoutUnits",
        "SimulatedLoss",
        "ClosingCarryover",
    ]
    result[numeric_output_columns] = result[numeric_output_columns].round(2)

    return result.sort_values(
        ["Model", "Date", "Store", "Product"]
    ).reset_index(drop=True)


def add_business_costs(
    simulation: pd.DataFrame,
    *,
    unit_cost_column: str = "UnitCost",
    unit_margin_column: str = "UnitMarginEstimate",
) -> pd.DataFrame:
    """Attach estimated yen values to a FIFO simulation.

    ``SalesMarginYen`` values fulfilled sales using the estimated unit margin.
    ``LossCostYen`` values expired units at their production cost.
    ``SalesMinusLossCostYen`` is the visible business result used by the hybrid
    notebook: sales margin minus the cost of expired products.

    ``StockoutCostYen`` and ``TotalBusinessCostYen`` are retained for backwards
    compatibility with the older Random-Forest notebook. Maximising
    ``SalesMinusLossCostYen`` gives the same buffer ranking as minimising
    ``TotalBusinessCostYen`` when actual demand is fixed.
    """
    required_columns = {
        "SimulatedLoss",
        "StockoutUnits",
        unit_cost_column,
        unit_margin_column,
    }
    missing_columns = required_columns.difference(simulation.columns)
    if missing_columns:
        raise KeyError(
            "Missing required cost columns: "
            + str(sorted(missing_columns))
        )

    result = simulation.copy()
    result[unit_cost_column] = pd.to_numeric(
        result[unit_cost_column],
        errors="coerce",
    )
    result[unit_margin_column] = pd.to_numeric(
        result[unit_margin_column],
        errors="coerce",
    )

    result["CostInformationComplete"] = (
        result[unit_cost_column].notna()
        & result[unit_margin_column].notna()
        & result[unit_cost_column].ge(0)
        & result[unit_margin_column].ge(0)
    )

    result["SalesMarginYen"] = np.where(
        result["CostInformationComplete"],
        result["FulfilledSales"] * result[unit_margin_column],
        np.nan,
    )
    result["LossCostYen"] = np.where(
        result["CostInformationComplete"],
        result["SimulatedLoss"] * result[unit_cost_column],
        np.nan,
    )
    result["SalesMinusLossCostYen"] = (
        result["SalesMarginYen"] - result["LossCostYen"]
    )

    # Kept for the older notebook. The hybrid notebook displays and selects
    # using SalesMinusLossCostYen instead.
    result["StockoutCostYen"] = np.where(
        result["CostInformationComplete"],
        result["StockoutUnits"] * result[unit_margin_column],
        np.nan,
    )
    result["TotalBusinessCostYen"] = (
        result["LossCostYen"] + result["StockoutCostYen"]
    )

    return result


def summarize_fifo_results(
    simulation: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("Model",),
) -> pd.DataFrame:
    """Aggregate FIFO and optional yen-cost results by strategy."""
    group_columns = list(group_columns)
    required_columns = {
        *group_columns,
        "ActualDemand",
        "ForecastDemand",
        "InitialCarryoverAdded",
        "RecommendedProduction",
        "FulfilledSales",
        "StockoutUnits",
        "SimulatedLoss",
        "ClosingCarryover",
        "SequenceStart",
    }
    missing_columns = required_columns.difference(simulation.columns)
    if missing_columns:
        raise KeyError(
            "Missing required summary columns: "
            + str(sorted(missing_columns))
        )

    rows: list[dict[str, object]] = []
    groupby_argument: str | list[str]
    groupby_argument = (
        group_columns[0]
        if len(group_columns) == 1
        else group_columns
    )

    for group_key, group in simulation.groupby(
        groupby_argument,
        sort=False,
        dropna=False,
    ):
        if len(group_columns) == 1:
            group_key = (group_key,)

        actual_demand = group["ActualDemand"].sum()
        fulfilled_sales = group["FulfilledSales"].sum()
        stockout_units = group["StockoutUnits"].sum()
        simulated_loss = group["SimulatedLoss"].sum()
        recommended_production = group["RecommendedProduction"].sum()
        initial_carryover = group["InitialCarryoverAdded"].sum()
        total_stock_entering_simulation = (
            recommended_production + initial_carryover
        )

        row = dict(zip(group_columns, group_key))
        row.update(
            {
                "Rows": len(group),
                "SequenceStarts": int(group["SequenceStart"].sum()),
                "ActualDemand": actual_demand,
                "ForecastDemand": group["ForecastDemand"].sum(),
                "RecommendedProduction": recommended_production,
                "FulfilledSales": fulfilled_sales,
                "ServiceLevel": (
                    fulfilled_sales / actual_demand
                    if actual_demand > 0
                    else np.nan
                ),
                "StockoutUnits": stockout_units,
                "StockoutRows": int(group["StockoutUnits"].gt(0).sum()),
                "StockoutRowRate": group["StockoutUnits"].gt(0).mean(),
                "SimulatedLoss": simulated_loss,
                "LossRows": int(group["SimulatedLoss"].gt(0).sum()),
                "LossRowRate": group["SimulatedLoss"].gt(0).mean(),
                "LossRate": (
                    simulated_loss / total_stock_entering_simulation
                    if total_stock_entering_simulation > 0
                    else np.nan
                ),
                "AverageClosingCarryover": group[
                    "ClosingCarryover"
                ].mean(),
                "MaximumClosingCarryover": group[
                    "ClosingCarryover"
                ].max(),
            }
        )

        if "CostInformationComplete" in group.columns:
            complete_cost_rows = group["CostInformationComplete"].fillna(False)
            row["CostRows"] = int(complete_cost_rows.sum())
            row["CostCoverage"] = complete_cost_rows.mean()

        for cost_column in [
            "SalesMarginYen",
            "LossCostYen",
            "SalesMinusLossCostYen",
            "StockoutCostYen",
            "TotalBusinessCostYen",
        ]:
            if cost_column in group.columns:
                row[cost_column] = group[cost_column].sum(min_count=1)

        rows.append(row)

    result = pd.DataFrame(rows)

    if "TotalBusinessCostYen" in result.columns:
        sort_columns = [
            "TotalBusinessCostYen",
            "StockoutUnits",
            "SimulatedLoss",
        ]
    else:
        sort_columns = ["StockoutUnits", "SimulatedLoss"]

    return result.sort_values(
        sort_columns,
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)
