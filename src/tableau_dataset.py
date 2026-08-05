from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import jpholiday
except ImportError:
    jpholiday = None

from src.data_preparation import (
    DEMO_SALES_RULES,
    add_excel_sales_and_loss,
    add_inventory_sales,
    add_product_type,
    load_production,
)


SYNTHETIC_STORE_FILES = {
    "Shop_A": "Shop_A_losses.xlsx",
    "Shop_B": "Shop_B_losses.xlsx",
    "Shop_C": "Shop_C_losses.xlsx",
}

WEEKDAY_ENGLISH = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

PRODUCTION_METHOD_CHANGE = pd.Timestamp("2026-06-27")
TABLEAU_OUTPUT_DIRECTORY = "tableau"

# Fallback names for the fixed public synthetic period. When jpholiday is
# installed, it remains the general source for dates outside this small range.
SYNTHETIC_JAPAN_HOLIDAYS = {
    pd.Timestamp("2026-05-03"): "Constitution Memorial Day",
    pd.Timestamp("2026-05-04"): "Greenery Day",
    pd.Timestamp("2026-05-05"): "Children's Day",
    pd.Timestamp("2026-05-06"): "Substitute Holiday",
    pd.Timestamp("2026-07-20"): "Marine Day",
}

# Fictional labels used only to make the public dashboard useful for filtering
# and comparison. They do not reproduce confidential company events.
SYNTHETIC_EVENTS = [
    {
        "Date": pd.Timestamp("2026-06-20"),
        "Store": "Shop_A",
        "EventName": "Synthetic shop anniversary",
        "PromotionType": "Half-price choux",
        "PromotionTarget": "シュー",
    },
    {
        "Date": pd.Timestamp("2026-07-11"),
        "Store": "All",
        "EventName": "Synthetic summer weekend campaign",
        "PromotionType": "Free choux with pastry purchase",
        "PromotionTarget": "シュー",
    },
]

# Operational fact-table columns. Calendar attributes deliberately live in the
# separate calendar.csv dimension table.
TABLEAU_FACT_COLUMNS = [
    "Date",
    "Store",
    "Product",
    "ProductType",
    "Forecast",
    "OpeningStock",
    "Production",
    "AvailableForSale",
    "Sales",
    "ClosingStock",
    "Loss",
    "ForecastError",
    "ForecastAbsoluteError",
    "SellThroughRate",
    "InventoryStatus",
    "StockoutFlag",
    "LossFlag",
    "ProductionMethod",
    "IsEventDay",
    "EventName",
    "PromotionType",
    "PromotionTarget",
    "SalesSource",
]

TABLEAU_CALENDAR_COLUMNS = [
    "Date",
    "Year",
    "Quarter",
    "MonthNumber",
    "MonthName",
    "WeekOfYear",
    "DayOfMonth",
    "WeekdayNumber",
    "Weekday",
    "IsWeekend",
    "IsHoliday",
    "HolidayName",
    "DayType",
    "Season",
]


def _add_synthetic_event_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Add clearly labelled fictional event metadata for dashboard practice."""
    result = data.copy()
    result["IsEventDay"] = False
    result["EventName"] = "None"
    result["PromotionType"] = "None"
    result["PromotionTarget"] = "None"

    for event in SYNTHETIC_EVENTS:
        mask = result["Date"].eq(event["Date"])
        if event["Store"] != "All":
            mask &= result["Store"].eq(event["Store"])

        result.loc[mask, "IsEventDay"] = True
        result.loc[mask, "EventName"] = event["EventName"]
        result.loc[mask, "PromotionType"] = event["PromotionType"]
        result.loc[mask, "PromotionTarget"] = event["PromotionTarget"]

    return result


def _season_from_month(month: pd.Series) -> pd.Series:
    """Return meteorological seasons for month numbers."""
    return pd.Series(
        np.select(
            [
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            ["Spring", "Summer", "Autumn"],
            default="Winter",
        ),
        index=month.index,
    )


def _holiday_name(value: pd.Timestamp) -> str:
    """Return a holiday name, with a deterministic fallback for demo dates."""
    normalized = pd.Timestamp(value).normalize()
    fallback = SYNTHETIC_JAPAN_HOLIDAYS.get(normalized)
    if fallback is not None:
        return fallback

    if jpholiday is None:
        return "None"

    name = jpholiday.is_holiday_name(normalized.date())
    return name if name else "None"


def build_tableau_calendar(date_values: pd.Series) -> pd.DataFrame:
    """Build one calendar-dimension row per unique date."""
    calendar = pd.DataFrame({"Date": pd.to_datetime(date_values).dt.normalize()})
    calendar = calendar.drop_duplicates().sort_values("Date").reset_index(drop=True)

    if calendar.empty:
        return pd.DataFrame(columns=TABLEAU_CALENDAR_COLUMNS)

    calendar["Year"] = calendar["Date"].dt.year.astype(int)
    calendar["Quarter"] = "Q" + calendar["Date"].dt.quarter.astype(str)
    calendar["MonthNumber"] = calendar["Date"].dt.month.astype(int)
    calendar["MonthName"] = calendar["Date"].dt.month_name()
    calendar["WeekOfYear"] = calendar["Date"].dt.isocalendar().week.astype(int)
    calendar["DayOfMonth"] = calendar["Date"].dt.day.astype(int)

    weekday_number = calendar["Date"].dt.dayofweek
    calendar["WeekdayNumber"] = (weekday_number + 1).astype(int)
    calendar["Weekday"] = weekday_number.map(WEEKDAY_ENGLISH)
    calendar["IsWeekend"] = weekday_number.ge(5)

    calendar["HolidayName"] = calendar["Date"].map(_holiday_name)
    calendar["IsHoliday"] = calendar["HolidayName"].ne("None")
    calendar["DayType"] = np.select(
        [calendar["IsHoliday"], calendar["IsWeekend"]],
        ["Holiday", "Weekend"],
        default="Weekday",
    )
    calendar["Season"] = _season_from_month(calendar["MonthNumber"])

    calendar = calendar[TABLEAU_CALENDAR_COLUMNS]

    if calendar["Date"].duplicated().any():
        raise ValueError("calendar.csv must contain one unique row per Date.")

    if calendar[TABLEAU_CALENDAR_COLUMNS].isna().any().any():
        null_columns = calendar.columns[calendar.isna().any()].tolist()
        raise ValueError(
            "Tableau calendar contains missing values in: "
            + ", ".join(null_columns)
        )

    return calendar


def build_tableau_dataset(project_root: str | Path) -> pd.DataFrame:
    """Build a clean operational fact table from the messy synthetic files.

    Calendar fields are intentionally excluded. Connect this fact table to
    ``calendar.csv`` in Tableau using a many-to-one relationship on ``Date``.
    """
    project_root = Path(project_root).resolve()
    raw_data_path = project_root / "data" / "synthetic_raw"
    production_file = raw_data_path / "synthetic_production.xlsx"

    required_files = [
        production_file,
        *(raw_data_path / name for name in SYNTHETIC_STORE_FILES.values()),
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing synthetic source files:\n"
            + "\n".join(str(path) for path in missing_files)
        )

    excel = pd.ExcelFile(production_file)
    sheet_names = excel.sheet_names[2:14]

    production = load_production(
        production_file=production_file,
        stores=list(SYNTHETIC_STORE_FILES),
        sheet_names=sheet_names,
        include_forecast=True,
    )
    production["ShopOpen"] = True
    production["ClosedReason"] = pd.NA
    production = add_product_type(production)
    production = add_excel_sales_and_loss(
        production=production,
        store_files=SYNTHETIC_STORE_FILES,
        base_path=raw_data_path,
        sales_rules=DEMO_SALES_RULES,
    )
    production = add_inventory_sales(production)

    # The first observation in each store-product series has no previous-day
    # closing stock, so sales cannot be reconstructed from inventory. Excluding
    # those rows gives Tableau complete measures without fabricated values.
    complete = production[
        [
            "Forecast",
            "CarryoverStock",
            "Production",
            "Sales",
            "ClosingStock",
            "Loss",
        ]
    ].notna().all(axis=1)
    data = production.loc[complete].copy()

    data = data.rename(
        columns={
            "Type": "ProductType",
            "CarryoverStock": "OpeningStock",
        }
    )

    quantity_columns = [
        "Forecast",
        "OpeningStock",
        "Production",
        "Sales",
        "ClosingStock",
        "Loss",
    ]
    data[quantity_columns] = data[quantity_columns].round().astype(int)

    data["AvailableForSale"] = (
        data["OpeningStock"] + data["Production"] - data["Loss"]
    )
    data["ForecastError"] = data["Forecast"] - data["Sales"]
    data["ForecastAbsoluteError"] = data["ForecastError"].abs()
    data["SellThroughRate"] = np.where(
        data["AvailableForSale"].gt(0),
        data["Sales"] / data["AvailableForSale"],
        0.0,
    ).round(4)

    data["InventoryStatus"] = np.select(
        [
            data["ClosingStock"].eq(0),
            data["ClosingStock"].le(5),
        ],
        ["Sold out", "Low closing stock"],
        default="Stock remaining",
    )
    data["StockoutFlag"] = data["ClosingStock"].eq(0)
    data["LossFlag"] = data["Loss"].gt(0)

    data["ProductionMethod"] = np.where(
        data["Date"].lt(PRODUCTION_METHOD_CHANGE),
        "Previous-day preparation",
        "Same-morning preparation",
    )
    data = _add_synthetic_event_fields(data)

    data = data[TABLEAU_FACT_COLUMNS].sort_values(
        ["Date", "Store", "Product"]
    ).reset_index(drop=True)

    duplicate_count = int(data.duplicated(["Date", "Store", "Product"]).sum())
    if duplicate_count:
        raise ValueError(
            f"Tableau dataset contains {duplicate_count} duplicate keys."
        )

    inventory_difference = (
        data["OpeningStock"]
        + data["Production"]
        - data["Loss"]
        - data["Sales"]
        - data["ClosingStock"]
    )
    if not inventory_difference.eq(0).all():
        raise ValueError("Inventory equation failed for the Tableau dataset.")

    if data[TABLEAU_FACT_COLUMNS].isna().any().any():
        null_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(
            "Tableau dataset contains missing values in: "
            + ", ".join(null_columns)
        )

    return data


def write_tableau_dataset(
    project_root: str | Path,
    output_file: str | Path | None = None,
) -> Path:
    """Create only ``tableau/bakery_tableau.csv`` and return its path."""
    project_root = Path(project_root).resolve()
    if output_file is None:
        output_file = project_root / TABLEAU_OUTPUT_DIRECTORY / "bakery_tableau.csv"
    else:
        output_file = Path(output_file).resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    data = build_tableau_dataset(project_root)
    data.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file


def write_tableau_exports(
    project_root: str | Path,
    fact_output_file: str | Path | None = None,
    calendar_output_file: str | Path | None = None,
) -> tuple[Path, Path]:
    """Create both public Tableau CSV files inside ``tableau/``."""
    project_root = Path(project_root).resolve()
    tableau_path = project_root / TABLEAU_OUTPUT_DIRECTORY

    fact_output = (
        Path(fact_output_file).resolve()
        if fact_output_file is not None
        else tableau_path / "bakery_tableau.csv"
    )
    calendar_output = (
        Path(calendar_output_file).resolve()
        if calendar_output_file is not None
        else tableau_path / "calendar.csv"
    )

    fact_output.parent.mkdir(parents=True, exist_ok=True)
    calendar_output.parent.mkdir(parents=True, exist_ok=True)

    fact = build_tableau_dataset(project_root)
    calendar = build_tableau_calendar(fact["Date"])

    fact.to_csv(fact_output, index=False, encoding="utf-8-sig")
    calendar.to_csv(calendar_output, index=False, encoding="utf-8-sig")

    missing_calendar_dates = set(fact["Date"].dt.normalize()) - set(calendar["Date"])
    if missing_calendar_dates:
        raise ValueError("Some fact-table dates are missing from calendar.csv.")

    return fact_output, calendar_output
