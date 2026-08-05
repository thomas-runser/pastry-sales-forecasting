from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REGULAR_PRODUCTS = [
    "シュー",
    "リッチシュー",
    "エクレア",
    "リッチエクレア",
    "パール",
    "アップルパイ",
    "プリン",
    "杏仁",
]

THREE_DAY_PRODUCTS = {
    "エクレア",
    "アップルパイ",
}


SOURCE_COLUMN_RENAMES = {
    "残在庫": "ClosingStock",
    "製造数": "Production",
    "予測": "Forecast",
}

FORM_SEASONAL_COLUMN = "季節シュー"

# Fictional prices and set amounts for the public synthetic demonstration.
# Private mode can replace them with local rules from
# ``private_config/sales_rules.py``.
DEMO_SALES_RULES = {
    "seasonal_price": 650,
    "standard_choux_price": 450,
    "yen_tolerance": 10,
    "set_amount_tolerance": 5,
    "direct_sales_form_columns": {
        "リッチシュー": ("リッチシュークリーム", 520),
        "エクレア": ("エクレア", 620),
    },
    "set_options": [
        # (recorded set revenue, standard choux quantity)
        (1200, 0),
        (1800, 0),
        (1500, 3),
        (2400, 5),
        (3600, 8),
    ],
}

LOSS_FORM_COLUMNS = {
    "シュー": "シュークリームロス",
    "リッチシュー": "リッチシューロス",
    "エクレア": "エクレアロス",
    "リッチエクレア": "リッチエクレアロス",
    "パール": "パールロス",
    "プリン": "プリンロス",
    "アップルパイ": "アップルパイロス",
}


# ---------------------------------------------------------------------------
# Production workbook loading
# ---------------------------------------------------------------------------

def clean_store(df: pd.DataFrame) -> pd.DataFrame:
    """Convert one store block from the wide Excel layout to tidy rows.

    Date headers are forward-filled, then only the first three columns for
    each date are used. Monthly summary columns are therefore ignored without
    checking their names.
    """
    dates = pd.to_datetime(
        df.iloc[0, 3:],
        errors="coerce",
    ).ffill()

    position_in_date = dates.groupby(dates).cumcount()
    keep_daily_column = dates.notna() & position_in_date.lt(3)

    column_numbers = (
        np.flatnonzero(keep_daily_column.to_numpy())
        + 3
    )
    daily_dates = dates.loc[keep_daily_column].reset_index(drop=True)
    daily_metrics = (
        df.iloc[1, column_numbers]
        .replace(["在庫", "在庫数"], "残在庫")
        .reset_index(drop=True)
    )
    products = df.iloc[2:, 2]
    frames = []

    for column_number, date, metric in zip(
        column_numbers,
        daily_dates,
        daily_metrics,
    ):
        if pd.isna(metric):
            continue

        frames.append(
            pd.DataFrame({
                "Date": date,
                "Product": products.to_numpy(),
                "Metric": str(metric).strip(),
                "Value": df.iloc[
                    2:,
                    column_number,
                ].to_numpy(),
            })
        )

    if not frames:
        return pd.DataFrame(columns=["Date", "Product"])

    clean = (
        pd.concat(frames, ignore_index=True)
        .pivot_table(
            index=["Date", "Product"],
            columns="Metric",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )
    clean.columns.name = None

    product_text = clean["Product"].astype("string").str.strip()
    clean = clean.loc[
        clean["Product"].notna()
        & product_text.ne("")
        & product_text.ne("合計")
    ].copy()

    value_columns = clean.columns.difference(["Date", "Product"])
    clean[value_columns] = clean[value_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    has_nonzero_value = (
        clean[value_columns]
        .ne(0)
        .fillna(False)
        .any(axis=1)
    )

    return clean.loc[has_nonzero_value].copy()


def split_stores(
    production: pd.DataFrame,
    stores: list[str],
) -> dict[str, pd.DataFrame]:
    """Split a worksheet into consecutive blocks, one for each store."""
    mask = production[0].isin(stores)
    groups = mask.cumsum()

    return {
        store: group
        for store, group in zip(
            stores,
            [group for _, group in production.groupby(groups)],
        )
    }


def load_production(
    production_file: str | Path,
    stores: list[str],
    sheet_names: list[str],
    include_forecast: bool = False,
) -> pd.DataFrame:
    """Load and clean monthly production worksheets.

    ``残在庫`` is the sellable stock remaining at the end of the recorded date,
    so it is returned as ``ClosingStock``. The workbook's ``Production`` value
    is not moved here; private mode aligns it to the following selling date with
    :func:`align_production_to_sales_date`. Set ``include_forecast=True`` only
    for public exports that need the workbook's synthetic ``Forecast`` field.
    """
    dfs = []

    for sheet_name in sheet_names:
        production = pd.read_excel(
            production_file,
            sheet_name=sheet_name,
            skiprows=14,
            header=None,
            parse_dates=True,
        )

        production.dropna(subset=[0, 1], how="all", inplace=True)
        production.dropna(axis=1, how="all", inplace=True)

        store_dfs = split_stores(production, stores)
        sheet_parts = []

        for store, store_df in store_dfs.items():
            cleaned = clean_store(store_df)
            cleaned["Store"] = store
            sheet_parts.append(cleaned)

        sheet_df = pd.concat(sheet_parts, ignore_index=True)

        main_month = sheet_df["Date"].dt.to_period("M").mode()[0]
        sheet_df = sheet_df.loc[
            sheet_df["Date"].dt.to_period("M").eq(main_month)
        ]
        dfs.append(sheet_df)

    production = pd.concat(dfs, ignore_index=True)
    if not include_forecast:
        production = production.drop(
            columns=["予測"],
            errors="ignore",
        )
    production = production.rename(columns=SOURCE_COLUMN_RENAMES)

    required_columns = {
        "Date",
        "Store",
        "Product",
        "ClosingStock",
        "Production",
    }
    missing_columns = required_columns.difference(production.columns)
    if missing_columns:
        raise KeyError(
            "Missing required production columns: "
            + str(sorted(missing_columns))
        )

    production["ClosingStock"] = pd.to_numeric(
        production["ClosingStock"],
        errors="coerce",
    ).round()
    production["Production"] = pd.to_numeric(
        production["Production"],
        errors="coerce",
    ).round()
    production["Date"] = pd.to_datetime(
        production["Date"]
    ).dt.normalize()

    return production.sort_values(
        ["Date", "Store", "Product"]
    ).reset_index(drop=True)


def align_production_to_sales_date(
    production: pd.DataFrame,
) -> pd.DataFrame:
    """Move every recorded production quantity to the following selling date.

    The workbook records August 1 production under August 1 even though those
    products are sold on August 2. ``ClosingStock`` remains on August 1 because
    it is the stock counted at the end of August 1.
    """
    data = production.copy()
    keys = ["Date", "Store", "Product"]

    required = {*keys, "ClosingStock", "Production"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(
            "Missing required production-alignment columns: "
            + str(sorted(missing))
        )

    closing_columns = [
        column
        for column in data.columns
        if column != "Production"
    ]
    closing_rows = data[closing_columns].copy()

    shifted = data[keys + ["Production"]].copy()
    shifted["Date"] += pd.Timedelta(days=1)
    shifted = (
        shifted.groupby(keys, as_index=False, dropna=False)["Production"]
        .sum(min_count=1)
    )

    result = closing_rows.merge(
        shifted,
        on=keys,
        how="outer",
        validate="one_to_one",
    )
    return result.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Business fields and loss
# ---------------------------------------------------------------------------

def add_product_type(production: pd.DataFrame) -> pd.DataFrame:
    """Label products as regular or seasonal."""
    production = production.copy()
    production["Type"] = (
        production["Product"]
        .isin(REGULAR_PRODUCTS)
        .map({True: "regular", False: "seasonal"})
    )
    return production


def _set_choux_sales_range(
    amount: float,
    set_options: list[tuple[int, int]],
    tolerance: int = 3,
) -> tuple[float, float]:
    """Return the minimum and maximum choux count represented by set sales.

    ``set_options`` contains ``(recorded revenue, standard choux quantity)``
    pairs. Passing the options explicitly keeps real business rules outside
    the public source code.
    """
    if pd.isna(amount) or float(amount) < 0:
        return np.nan, np.nan

    target = int(round(float(amount)))
    limit = target + tolerance
    minimum = [None] * (limit + 1)
    maximum = [None] * (limit + 1)
    minimum[0] = 0
    maximum[0] = 0

    for total in range(limit + 1):
        if minimum[total] is None:
            continue

        for price, choux_count in set_options:
            new_total = total + int(price)
            if new_total > limit:
                continue

            low = minimum[total] + int(choux_count)
            high = maximum[total] + int(choux_count)

            if minimum[new_total] is None:
                minimum[new_total] = low
                maximum[new_total] = high
            else:
                minimum[new_total] = min(minimum[new_total], low)
                maximum[new_total] = max(maximum[new_total], high)

    possible = [
        total
        for total in range(max(0, target - tolerance), limit + 1)
        if minimum[total] is not None
    ]
    if not possible:
        return np.nan, np.nan

    return (
        min(minimum[total] for total in possible),
        max(maximum[total] for total in possible),
    )


def add_excel_sales_and_loss(
    production: pd.DataFrame,
    store_files: dict[str, str],
    base_path: str | Path,
    sales_rules: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Copy loss and recorded sales from each shop form onto matching rows.

    Loss is copied directly to the same Date/Store/Product row. Sales amounts
    are converted from yen to units using configurable pricing rules. Only
    standard choux needs a minimum/maximum range because the form combines
    several sets. ``客数`` is not read.
    """
    rules = DEMO_SALES_RULES if sales_rules is None else sales_rules

    seasonal_price = float(rules["seasonal_price"])
    standard_choux_price = float(rules["standard_choux_price"])
    yen_tolerance = float(rules.get("yen_tolerance", 10))
    set_amount_tolerance = int(rules.get("set_amount_tolerance", 3))
    direct_sales_form_columns = dict(rules["direct_sales_form_columns"])
    set_options = [
        (int(price), int(quantity))
        for price, quantity in rules["set_options"]
    ]

    data = production.copy()
    data["Loss"] = np.nan
    data["Sales"] = np.nan
    data["SalesSource"] = pd.NA
    data["ExcelSalesMin"] = np.nan
    data["ExcelSalesMax"] = np.nan

    form_columns = {
        "日付",
        "シュークリーム",
        "季節シュー",
        "セット",
        "季節シューロス",
        *LOSS_FORM_COLUMNS.values(),
        *(column for column, _ in direct_sales_form_columns.values()),
    }

    for store, file_name in store_files.items():
        form = pd.read_excel(
            Path(base_path) / file_name,
            sheet_name="フォームの回答 1",
            usecols=lambda column: column in form_columns,
        )
        form["日付"] = pd.to_datetime(
            form["日付"],
            errors="coerce",
        ).dt.normalize()
        form = (
            form.dropna(subset=["日付"])
            .drop_duplicates("日付", keep="last")
            .set_index("日付")
        )
        form = form.apply(pd.to_numeric, errors="coerce")
        store_mask = data["Store"].eq(store)

        # Loss: copy the Excel quantity directly to the same dated product row.
        for product, column in LOSS_FORM_COLUMNS.items():
            if column not in form:
                continue
            mask = store_mask & data["Product"].eq(product)
            data.loc[mask, "Loss"] = data.loc[mask, "Date"].map(
                form[column].fillna(0)
            )

        seasonal_rows = data.loc[
            store_mask & data["Type"].eq("seasonal")
        ]
        active_seasonal = seasonal_rows.loc[
            seasonal_rows[["Production", "ClosingStock"]]
            .fillna(0)
            .ne(0)
            .any(axis=1)
        ]
        one_active = active_seasonal.groupby("Date")["Product"].transform(
            "size"
        ).eq(1)
        active_seasonal = active_seasonal.loc[one_active]

        if "季節シューロス" in form:
            data.loc[active_seasonal.index, "Loss"] = (
                active_seasonal["Date"]
                .map(form["季節シューロス"].fillna(0))
                .to_numpy()
            )

        # Products recorded directly as yen amounts.
        for product, (column, price) in direct_sales_form_columns.items():
            if column not in form:
                continue
            amount = form[column].fillna(0)
            units = amount.div(price).round()
            units = units.mask(
                amount.sub(units.mul(price)).abs().gt(yen_tolerance)
            )
            mask = store_mask & data["Product"].eq(product)
            data.loc[mask, "Sales"] = data.loc[mask, "Date"].map(units)

        # The generic seasonal column belongs to the one active seasonal row.
        if "季節シュー" in form:
            amount = form["季節シュー"].fillna(0)
            units = amount.div(seasonal_price).round()
            units = units.mask(
                amount.sub(units.mul(seasonal_price))
                .abs()
                .gt(yen_tolerance)
            )
            data.loc[active_seasonal.index, "Sales"] = (
                active_seasonal["Date"].map(units).to_numpy()
            )

        # Standard choux = direct choux revenue + choux contained in sets.
        if "シュークリーム" in form and "セット" in form:
            mask = store_mask & data["Product"].eq("シュー")
            rows = data.loc[mask]

            direct_amount = rows["Date"].map(form["シュークリーム"].fillna(0))
            discount = pd.to_numeric(
                rows.get("DiscountRate", pd.Series(0, index=rows.index)),
                errors="coerce",
            ).fillna(0)
            price = standard_choux_price * (1 - discount)
            direct_units = direct_amount.div(price).round()
            direct_units = direct_units.mask(
                direct_amount.sub(direct_units.mul(price))
                .abs()
                .gt(yen_tolerance)
            )

            set_amount = rows["Date"].map(form["セット"].fillna(0))
            set_range = set_amount.apply(
                lambda amount: _set_choux_sales_range(
                    amount=amount,
                    set_options=set_options,
                    tolerance=set_amount_tolerance,
                )
            )
            minimum = direct_units + set_range.str[0]
            maximum = direct_units + set_range.str[1]

            data.loc[mask, "ExcelSalesMin"] = minimum.to_numpy()
            data.loc[mask, "ExcelSalesMax"] = maximum.to_numpy()

            exact = minimum.notna() & minimum.eq(maximum)
            data.loc[rows.index[exact], "Sales"] = minimum.loc[exact].to_numpy()

    data["Loss"] = pd.to_numeric(data["Loss"], errors="coerce").round()
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce").round()
    data.loc[data["Sales"].notna(), "SalesSource"] = "excel"
    return data


def add_inventory_sales(production: pd.DataFrame) -> pd.DataFrame:
    """Calculate inventory sales, fill missing Sales, and compare with Excel."""
    data = production.sort_values(
        ["Store", "Product", "Date"]
    ).copy()

    numeric_columns = ["ClosingStock", "Production", "Loss", "Sales"]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    groups = data.groupby(["Store", "Product"], sort=False)
    previous_date = groups["Date"].shift(1)
    previous_closing = groups["ClosingStock"].shift(1)
    previous_day = (data["Date"] - previous_date).eq(pd.Timedelta(days=1))
    data["CarryoverStock"] = previous_closing.where(previous_day)

    complete = data[
        ["CarryoverStock", "Production", "Loss", "ClosingStock"]
    ].notna().all(axis=1)
    data["InventorySales"] = np.nan
    data.loc[complete, "InventorySales"] = (
        data.loc[complete, "CarryoverStock"]
        + data.loc[complete, "Production"]
        - data.loc[complete, "Loss"]
        - data.loc[complete, "ClosingStock"]
    ).round()

    data["ExcelSales"] = data["Sales"]

    missing = data["Sales"].isna() & data["InventorySales"].ge(0)
    data.loc[missing, "Sales"] = data.loc[missing, "InventorySales"]
    data.loc[missing, "SalesSource"] = "inventory"

    return data


# ---------------------------------------------------------------------------
# FIFO stock-age diagnostics
# ---------------------------------------------------------------------------

def add_stock_age(production: pd.DataFrame) -> pd.DataFrame:
    """Estimate FIFO stock age and store all issues in one text column."""
    production = production.sort_values(
        ["Store", "Product", "Date"]
    ).copy()

    required = {
        "CarryoverStock",
        "ClosingStock",
        "Production",
        "Loss",
        "Sales",
    }
    missing = required.difference(production.columns)
    if missing:
        raise KeyError(
            "Missing stock-age columns: "
            + str(sorted(missing))
        )

    carryover = pd.to_numeric(
        production["CarryoverStock"],
        errors="coerce",
    ).astype(float)
    previous_production = pd.to_numeric(
        production.groupby(
            ["Store", "Product"],
            sort=False,
        )["Production"].shift(1),
        errors="coerce",
    )

    production["StockD1"] = carryover
    production["StockD2"] = 0.0

    three_day = production["Product"].isin(THREE_DAY_PRODUCTS)
    inferred_d1 = np.minimum(
        carryover[three_day],
        previous_production[three_day].fillna(carryover[three_day]),
    )
    production.loc[three_day, "StockD1"] = inferred_d1
    production.loc[three_day, "StockD2"] = (
        carryover[three_day] - inferred_d1
    ).clip(lower=0)

    sales = pd.to_numeric(production["Sales"], errors="coerce")
    stock_d1 = pd.to_numeric(production["StockD1"], errors="coerce")
    stock_d2 = pd.to_numeric(production["StockD2"], errors="coerce")
    production_qty = pd.to_numeric(
        production["Production"],
        errors="coerce",
    )

    production["StockD2Sold"] = np.minimum(
        sales.clip(lower=0),
        stock_d2,
    )
    remaining_sales = (
        sales - production["StockD2Sold"]
    ).clip(lower=0)
    production["StockD1Sold"] = np.minimum(
        remaining_sales,
        stock_d1,
    )
    production["ProductionSold"] = (
        remaining_sales - production["StockD1Sold"]
    ).clip(lower=0)

    production["ExpectedLoss"] = (
        stock_d1 - production["StockD1Sold"]
    ).clip(lower=0)
    production.loc[three_day, "ExpectedLoss"] = (
        stock_d2[three_day]
        - production.loc[three_day, "StockD2Sold"]
    ).clip(lower=0)

    production["ExpectedClosingStock"] = (
        production_qty - production["ProductionSold"]
    )
    production.loc[three_day, "ExpectedClosingStock"] += (
        stock_d1[three_day]
        - production.loc[three_day, "StockD1Sold"]
    ).clip(lower=0)

    inventory_balance = (
        production["CarryoverStock"]
        + production["Production"]
        - production["Loss"]
        - production["ClosingStock"]
        - production["Sales"]
    )

    def diagnose(row: pd.Series) -> object:
        """Return one readable sentence list for a prepared row."""
        messages = []

        def number(value: object) -> str:
            return f"{float(value):g}"

        if pd.notna(row["Production"]) and row["Production"] < 0:
            messages.append(
                f"Negative production ({number(row['Production'])})"
            )
        if pd.notna(row["Sales"]) and row["Sales"] < 0:
            messages.append(f"Negative sales ({number(row['Sales'])})")

        if (
            row["Product"] == "シュー"
            and pd.notna(row.get("ExcelSalesMin"))
            and pd.notna(row.get("ExcelSalesMax"))
            and pd.notna(row.get("InventorySales"))
            and not (
                row["ExcelSalesMin"]
                <= row["InventorySales"]
                <= row["ExcelSalesMax"]
            )
        ):
            messages.append(
                "Inventory sales "
                f"{number(row['InventorySales'])} outside Excel choux range "
                f"{number(row['ExcelSalesMin'])}–{number(row['ExcelSalesMax'])}"
            )
        elif (
            pd.notna(row.get("ExcelSales"))
            and pd.notna(row.get("InventorySales"))
            and row["ExcelSales"] != row["InventorySales"]
        ):
            messages.append(
                f"Excel sales {number(row['ExcelSales'])} differs from "
                f"inventory sales {number(row['InventorySales'])}"
            )

        if (
            pd.notna(row["Loss"])
            and pd.notna(row["CarryoverStock"])
            and row["Loss"] > row["CarryoverStock"]
        ):
            messages.append(
                f"Loss {number(row['Loss'])} exceeds carryover stock "
                f"{number(row['CarryoverStock'])}"
            )

        if (
            pd.notna(row["Loss"])
            and pd.notna(row["ExpectedLoss"])
            and row["Loss"] != row["ExpectedLoss"]
        ):
            messages.append(
                f"Loss {number(row['Loss'])} differs from FIFO expected "
                f"{number(row['ExpectedLoss'])}"
            )

        if (
            pd.notna(row["ClosingStock"])
            and pd.notna(row["ExpectedClosingStock"])
            and row["ClosingStock"] != row["ExpectedClosingStock"]
        ):
            messages.append(
                f"Closing stock {number(row['ClosingStock'])} differs from "
                f"FIFO expected {number(row['ExpectedClosingStock'])}"
            )

        if (
            pd.notna(row["ProductionSold"])
            and pd.notna(row["Production"])
            and row["ProductionSold"] > row["Production"]
        ):
            messages.append(
                f"FIFO uses {number(row['ProductionSold'])} new units but "
                f"production is {number(row['Production'])}"
            )

        if pd.notna(row["InventoryBalance"]) and row["InventoryBalance"] != 0:
            messages.append(
                "Inventory does not balance "
                f"({number(row['InventoryBalance'])})"
            )

        return "; ".join(messages) if messages else pd.NA

    diagnostic_rows = production.copy()
    diagnostic_rows["InventoryBalance"] = inventory_balance
    production["Diagnostics"] = diagnostic_rows.apply(diagnose, axis=1)

    return production


REQUIRED_MODEL_COLUMNS = [
    "Date",
    "Store",
    "Product",
    "Sales",
]


def mark_usable_rows(production: pd.DataFrame) -> pd.DataFrame:
    """Create separate flags for demand and inventory modelling."""
    production = production.copy()

    has_required_values = production[REQUIRED_MODEL_COLUMNS].notna().all(axis=1)
    reliable_sales = (
        production["Sales"].notna()
        & production["Sales"].ge(0)
    )

    production["UseForSalesModel"] = (
        production["ShopOpen"].fillna(False)
        & has_required_values
        & reliable_sales
    )

    inventory_values = production[
        [
            "CarryoverStock",
            "Production",
            "Loss",
            "ClosingStock",
        ]
    ].notna().all(axis=1)

    inventory_balance = (
        production["CarryoverStock"]
        + production["Production"]
        - production["Loss"]
        - production["ClosingStock"]
        - production["Sales"]
    )
    critical_inventory_issue = (
        production["Loss"].gt(production["CarryoverStock"])
        | production["ProductionSold"].gt(production["Production"])
        | production["ExpectedClosingStock"].lt(0)
        | inventory_balance.abs().gt(0)
    ).fillna(False)

    production["UseForInventoryModel"] = (
        production["UseForSalesModel"]
        & inventory_values
        & ~critical_inventory_issue
    )

    return production
