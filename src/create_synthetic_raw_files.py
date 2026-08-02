from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STORES = ["Shop_A", "Shop_B", "Shop_C"]

PRODUCT_CONFIG = {
    "シュー": {
        "base_demand": 28,
        "loss_column": "シュークリームロス",
    },
    "リッチシュー": {
        "base_demand": 15,
        "loss_column": "リッチシューロス",
    },
    "エクレア": {
        "base_demand": 12,
        "loss_column": "エクレアロス",
    },
    "リッチエクレア": {
        "base_demand": 8,
        "loss_column": "リッチエクレアロス",
    },
    "パール": {
        "base_demand": 9,
        "loss_column": "パールロス",
    },
    "プリン": {
        "base_demand": 10,
        "loss_column": "プリンロス",
    },
    "アップルパイ": {
        "base_demand": 7,
        "loss_column": "アップルパイロス",
    },
    "季節シュー": {
        "base_demand": 11,
        "loss_column": "季節シューロス",
    },
}

PRODUCTS = list(PRODUCT_CONFIG)

ALL_LOSS_COLUMNS = list(
    dict.fromkeys(
        config["loss_column"]
        for config in PRODUCT_CONFIG.values()
    )
)

METRICS = ["予測", "残在庫", "製造数"]

STORE_FACTOR = {
    "Shop_A": 1.00,
    "Shop_B": 0.82,
    "Shop_C": 1.15,
}

APPLE_PIE_LAUNCH = pd.Timestamp("2026-07-09")


def generate_synthetic_records(
    start_date: str,
    end_date: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Generate tidy synthetic production, closing stock, sales, and losses."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")

    if dates.empty:
        raise ValueError("end_date must be on or after start_date.")

    records: list[dict] = []
    truth: list[dict] = []

    losses_by_store = {
        store: pd.DataFrame(
            0,
            index=dates,
            columns=ALL_LOSS_COLUMNS,
            dtype=int,
        )
        for store in STORES
    }

    for store in STORES:
        for product in PRODUCTS:
            carryover_stock = (
                0
                if product == "アップルパイ"
                else int(rng.integers(2, 9))
            )

            for day_number, date in enumerate(dates):
                before_launch = (
                    product == "アップルパイ"
                    and date < APPLE_PIE_LAUNCH
                )

                if before_launch:
                    forecast = production = loss = sales = closing_stock = 0
                else:
                    weekend_factor = 1.20 if date.dayofweek >= 5 else 1.0
                    smooth_seasonality = (
                        1
                        + 0.08
                        * np.sin(2 * np.pi * day_number / 14)
                    )

                    expected = (
                        PRODUCT_CONFIG[product]["base_demand"]
                        * STORE_FACTOR[store]
                        * weekend_factor
                        * smooth_seasonality
                    )

                    forecast = max(
                        0,
                        int(round(expected + rng.normal(0, 1.5))),
                    )

                    desired_closing_stock = max(
                        1,
                        int(round(forecast * 0.12)),
                    )

                    production = max(
                        0,
                        forecast
                        + desired_closing_stock
                        - carryover_stock,
                    )

                    loss = int(rng.random() < 0.04)
                    # Recorded shop loss can only come from carried-over stock.
                    # Production waste is not part of the daily shop-loss form.
                    loss = min(
                        loss,
                        carryover_stock,
                    )

                    available = carryover_stock + production - loss

                    demand = max(
                        0,
                        int(round(expected + rng.normal(0, 2.0))),
                    )

                    sales = min(demand, available)
                    closing_stock = available - sales

                records.append(
                    {
                        "Date": date,
                        "Store": store,
                        "Product": product,
                        "予測": forecast,
                        # 残在庫 is the stock remaining at the end of this date.
                        "残在庫": closing_stock,
                        "製造数": production,
                    }
                )

                truth.append(
                    {
                        "Date": date,
                        "Store": store,
                        "Product": product,
                        "ExpectedSales": sales,
                        "ExpectedLoss": loss,
                        "ExpectedClosingStock": closing_stock,
                    }
                )

                loss_column = PRODUCT_CONFIG[product]["loss_column"]
                losses_by_store[store].loc[date, loss_column] = loss
                carryover_stock = closing_stock

    return (
        pd.DataFrame(records),
        pd.DataFrame(truth),
        losses_by_store,
        dates,
    )

def apply_synthetic_anomalies(
    raw: pd.DataFrame,
    include_anomalies: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    """Optionally add known demonstration errors to the synthetic data."""
    raw = raw.copy()
    anomaly_log: list[dict] = []

    if not include_anomalies:
        return raw, anomaly_log

    missing_mask = (
        raw["Store"].eq("Shop_B")
        & raw["Product"].eq("シュー")
        & raw["Date"].eq(pd.Timestamp("2026-06-12"))
    )

    if missing_mask.any():
        raw.loc[missing_mask, "残在庫"] = np.nan
        anomaly_log.append(
            {
                "Date": "2026-06-12",
                "Store": "Shop_B",
                "Product": "シュー",
                "Field": "残在庫",
                "Change": "set to missing",
            }
        )

    rounding_mask = (
        raw["Store"].eq("Shop_A")
        & raw["Product"].eq("リッチシュー")
        & raw["Date"].eq(pd.Timestamp("2026-06-18"))
    )

    if rounding_mask.any():
        raw.loc[rounding_mask, "製造数"] = (
            raw.loc[rounding_mask, "製造数"] + 1
        )

        anomaly_log.append(
            {
                "Date": "2026-06-18",
                "Store": "Shop_A",
                "Product": "リッチシュー",
                "Field": "製造数",
                "Change": "+1 box-rounding difference",
            }
        )

    return raw, anomaly_log


def build_production_sheet(
    month_data: pd.DataFrame,
) -> pd.DataFrame:
    """Convert tidy records to the horizontal raw workbook layout."""
    month_dates = sorted(month_data["Date"].unique())
    width = 3 + len(month_dates) * len(METRICS)
    rows: list[list[object]] = []

    for store in STORES:
        store_rows = month_data[
            month_data["Store"].eq(store)
        ]

        date_row: list[object] = [store, None, None]
        metric_row: list[object] = [None, "Metric", None]

        for date in month_dates:
            date_row.extend([pd.Timestamp(date), None, None])
            metric_row.extend(METRICS)

        rows.extend([date_row, metric_row])

        for product_number, product in enumerate(
            PRODUCTS,
            start=1,
        ):
            product_row: list[object] = [
                None,
                product_number,
                product,
            ]

            product_data = (
                store_rows[
                    store_rows["Product"].eq(product)
                ]
                .set_index("Date")
            )

            for date in month_dates:
                values = product_data.loc[
                    pd.Timestamp(date),
                    METRICS,
                ]
                product_row.extend(values.tolist())

            rows.append(product_row)

    return pd.DataFrame(rows, columns=range(width))


def write_production_workbook(
    raw: pd.DataFrame,
    output_dir: Path,
    dates: pd.DatetimeIndex,
    seed: int,
    start_date: str,
    end_date: str,
) -> tuple[Path, list[str]]:
    """Write the synthetic monthly production workbook."""
    production_file = output_dir / "synthetic_production.xlsx"

    month_periods = pd.period_range(
        dates.min().to_period("M"),
        dates.max().to_period("M"),
        freq="M",
    )

    sheet_names = [str(period) for period in month_periods]

    with pd.ExcelWriter(
        production_file,
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(
            {
                "Synthetic demo": [
                    "This workbook contains generated data only.",
                    "It is structurally compatible with load_production().",
                ]
            }
        ).to_excel(
            writer,
            sheet_name="README",
            index=False,
        )

        pd.DataFrame(
            {
                "Setting": [
                    "Seed",
                    "Start date",
                    "End date",
                    "Stores",
                ],
                "Value": [
                    seed,
                    start_date,
                    end_date,
                    ", ".join(STORES),
                ],
            }
        ).to_excel(
            writer,
            sheet_name="Metadata",
            index=False,
        )

        for period, sheet_name in zip(
            month_periods,
            sheet_names,
        ):
            month_data = raw[
                raw["Date"].dt.to_period("M").eq(period)
            ]

            sheet = build_production_sheet(month_data)

            sheet.to_excel(
                writer,
                sheet_name=sheet_name,
                startrow=14,
                header=False,
                index=False,
            )

    return production_file, sheet_names


def write_loss_workbooks(
    losses_by_store: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, str]:
    """Write one daily-loss workbook per synthetic shop."""
    store_paths: dict[str, str] = {}

    for store, loss_data in losses_by_store.items():
        loss_file_name = f"{store}_losses.xlsx"
        store_paths[store] = loss_file_name

        form = loss_data.reset_index(names="日付")

        with pd.ExcelWriter(
            output_dir / loss_file_name,
            engine="openpyxl",
        ) as writer:
            form.to_excel(
                writer,
                sheet_name="フォームの回答 1",
                index=False,
            )

    return store_paths


def write_anomaly_log(
    anomaly_log: list[dict],
    output_dir: Path,
) -> None:
    """Write or remove the anomaly log so stale files are not retained."""
    anomaly_file = output_dir / "synthetic_anomalies.csv"

    if anomaly_log:
        pd.DataFrame(anomaly_log).to_csv(
            anomaly_file,
            index=False,
            encoding="utf-8-sig",
        )
    elif anomaly_file.exists():
        anomaly_file.unlink()


def create_synthetic_raw_files(
    output_dir: str | Path,
    start_date: str = "2026-05-01",
    end_date: str = "2026-07-31",
    seed: int = 42,
    include_anomalies: bool = False,
) -> dict:
    """
    Create synthetic Excel files that reproduce the structure of the original
    work files without including confidential business data.

    This function was developed with AI assistance and preserves the expected
    workbook format so the rest of the data-cleaning pipeline can be demonstrated
    using synthetic data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, truth_df, losses_by_store, dates = generate_synthetic_records(
        start_date=start_date,
        end_date=end_date,
        seed=seed,
    )

    raw, anomaly_log = apply_synthetic_anomalies(
        raw,
        include_anomalies=include_anomalies,
    )

    production_file, sheet_names = write_production_workbook(
        raw=raw,
        output_dir=output_dir,
        dates=dates,
        seed=seed,
        start_date=start_date,
        end_date=end_date,
    )

    store_paths = write_loss_workbooks(
        losses_by_store=losses_by_store,
        output_dir=output_dir,
    )

    truth_file = output_dir / "synthetic_expected_sales.csv"
    truth_df.to_csv(
        truth_file,
        index=False,
        encoding="utf-8-sig",
    )

    write_anomaly_log(
        anomaly_log=anomaly_log,
        output_dir=output_dir,
    )

    return {
        "data_path": str(output_dir.resolve()) + "/",
        "production_file": str(production_file.resolve()),
        "stores": STORES.copy(),
        "store_paths": store_paths,
        "sheet_names": sheet_names,
        "truth_file": str(truth_file.resolve()),
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    config = create_synthetic_raw_files(
        output_dir=project_root / "data" / "synthetic_raw",
        start_date="2026-05-01",
        end_date="2026-07-31",
        seed=42,
        include_anomalies=False,
    )

    print("Synthetic files created:")
    for key, value in config.items():
        print(f"- {key}: {value}")
