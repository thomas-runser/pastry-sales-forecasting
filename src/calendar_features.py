from __future__ import annotations

try:
    import jpholiday
except ImportError:
    # Calendar generation can still run with weekend/weekday features when the
    # optional Japanese-holiday package has not been installed yet.
    jpholiday = None

import pandas as pd
import requests


weekdays = {
    0: "月曜日",
    1: "火曜日",
    2: "水曜日",
    3: "木曜日",
    4: "金曜日",
    5: "土曜日",
    6: "日曜日",
}

weather_names = {
    0: "Clear",
    1: "Mostly clear",
    2: "Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    61: "Rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy showers",
    95: "Thunderstorm",
    96: "Thunderstorm",
    99: "Thunderstorm",
}

WEATHER_COLUMNS = [
    "Weather",
    "TemperatureMin",
    "TemperatureMax",
    "TemperatureAvg",
]


def create_calendar(production):
    """
    Create public calendar features.

    This function does not contain a private location or download weather.
    """
    calendar = (
        production[["Date"]]
        .drop_duplicates()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if calendar.empty:
        return pd.DataFrame(
            columns=["Date", "Weekday", "DayType"]
        )

    calendar["Date"] = (
        pd.to_datetime(calendar["Date"])
        .dt.normalize()
    )

    calendar["Weekday"] = (
        calendar["Date"]
        .dt.dayofweek
        .map(weekdays)
    )

    if jpholiday is not None:
        holiday_mask = (
            calendar["Date"]
            .dt.date
            .map(jpholiday.is_holiday_name)
            .notna()
        )
    else:
        # Keep a Boolean Series aligned to the calendar index. The notebook can
        # be executed before installing jpholiday, but Japanese public holidays
        # will temporarily remain classified as ordinary weekdays/weekends.
        holiday_mask = pd.Series(False, index=calendar.index)

    calendar["DayType"] = "平日"

    calendar.loc[
        calendar["Weekday"].isin(["土曜日", "日曜日"]),
        "DayType",
    ] = "週末"

    calendar.loc[
        holiday_mask,
        "DayType",
    ] = "休日"

    return calendar


def add_historical_weather(
    calendar,
    latitude,
    longitude,
    timezone="Asia/Tokyo",
):
    """
    Add historical weather using coordinates supplied by the caller.
    """
    
    calendar = calendar.copy()

    if calendar.empty:
        for column in WEATHER_COLUMNS:
            calendar[column] = pd.NA

        return calendar

    start_date = calendar["Date"].min()

    today = (
        pd.Timestamp.now(tz=timezone)
        .tz_localize(None)
        .normalize()
    )

    archive_end = min(
        calendar["Date"].max(),
        today - pd.Timedelta(days=1),
    )

    # There is no historical weather for future-only data.
    if archive_end < start_date:
        for column in WEATHER_COLUMNS:
            calendar[column] = pd.NA

        return calendar

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": archive_end.strftime("%Y-%m-%d"),
        "daily": (
            "weather_code,"
            "temperature_2m_min,"
            "temperature_2m_max,"
            "temperature_2m_mean"
        ),
        "timezone": timezone,
    }

    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            "Historical weather could not be downloaded."
        ) from error

    if "daily" not in data:
        raise ValueError(
            data.get(
                "reason",
                "Weather data was not returned",
            )
        )

    weather = pd.DataFrame({
        "Date": pd.to_datetime(
            data["daily"]["time"]
        ),
        "WeatherCode": data["daily"]["weather_code"],
        "TemperatureMin": data["daily"]["temperature_2m_min"],
        "TemperatureMax": data["daily"]["temperature_2m_max"],
        "TemperatureAvg": data["daily"]["temperature_2m_mean"],
    })

    weather["Weather"] = (
        weather["WeatherCode"]
        .map(weather_names)
    )

    temperature_columns = [
        "TemperatureMin",
        "TemperatureMax",
        "TemperatureAvg",
    ]

    weather[temperature_columns] = (
        weather[temperature_columns]
        .round(1)
    )

    return calendar.merge(
        weather[
            [
                "Date",
                "Weather",
                "TemperatureMin",
                "TemperatureMax",
                "TemperatureAvg",
            ]
        ],
        on="Date",
        how="left",
    )