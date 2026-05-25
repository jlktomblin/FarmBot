import pandas as pd
from datetime import datetime
import requests
import io
import time
import sys

# =========================================================
# BENDERBROOK HISTORICAL RAINFALL ANALYSIS
# =========================================================
#
# PURPOSE:
# Pull 10 years of Environment Canada precipitation data
# for a nearby weather station and calculate:
#
# 1. Weekly rainfall totals
# 2. 10-year historical weekly averages
# 3. Seasonal rainfall summaries
# 4. Weekly prediction based on historical averages
#
# =========================================================

# =========================================================
# CONFIG
# =========================================================

# Recommended nearby stations for Waterloo/Wellington region
#
# You can switch stations easily by changing TARGET_STATION
#
STATIONS = {
    "Waterloo_Wellington_A": 48549,
    "Elora": 3021920,
    "Stratford": 3057165,
    "Mount_Forest": 6114900
}

TARGET_STATION = "Waterloo_Wellington_A"
STATION_ID = STATIONS[TARGET_STATION]

START_YEAR = datetime.now().year - 10
END_YEAR = datetime.now().year - 1

# Growing season
START_MONTH = 5
END_MONTH = 9

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

# =========================================================
# STORAGE
# =========================================================

all_data = []

# =========================================================
# DOWNLOAD HISTORICAL WEATHER DATA
# =========================================================

for year in range(START_YEAR, END_YEAR + 1):

    print(f"\n=================================================")
    print(f"Pulling {year} data from {TARGET_STATION}")
    print("=================================================")

    url = (
        "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
        f"format=csv"
        f"&stationID={STATION_ID}"
        f"&Year={year}"
        f"&Month=6"
        f"&Day=1"
        f"&timeframe=2"
        f"&submit=Download+Data"
    )

    try:

        response = requests.get(url, headers=HEADERS, timeout=30)

        # -------------------------------------------------
        # CHECK FOR BAD RESPONSE
        # -------------------------------------------------

        if response.status_code != 200:
            print(f"FAILED: HTTP {response.status_code}")
            continue

        # Sometimes Environment Canada returns HTML pages
        if "<html" in response.text[:300].lower():
            print("FAILED: HTML page returned instead of CSV")
            continue

        # -------------------------------------------------
        # FIND START OF CSV TABLE
        # -------------------------------------------------

        lines = response.text.split("\n")

        start_line = None

        for i, line in enumerate(lines):

            if "Date/Time" in line:
                start_line = i
                break

        if start_line is None:
            print("FAILED: Could not find CSV header")
            continue

        # -------------------------------------------------
        # LOAD CSV
        # -------------------------------------------------

        df = pd.read_csv(
            io.StringIO("\n".join(lines[start_line:])),
            low_memory=False
        )

        print("\nDetected Columns:")
        print(df.columns.tolist())

        # -------------------------------------------------
        # AUTO-DETECT PRECIP COLUMN
        # -------------------------------------------------

        possible_precip_columns = [
            "Total Precip (mm)",
            "Total Rain (mm)",
            "Precip. Amount (mm)"
        ]

        precip_col = None

        for col in possible_precip_columns:

            if col in df.columns:
                precip_col = col
                break

        if precip_col is None:
            print("FAILED: No precipitation column found")
            continue

        print(f"\nUsing precipitation column: {precip_col}")

        # -------------------------------------------------
        # KEEP ONLY REQUIRED COLUMNS
        # -------------------------------------------------

        df = df[["Date/Time", precip_col]].copy()

        df.columns = ["date", "rain_mm"]

        # -------------------------------------------------
        # DATE CONVERSION
        # -------------------------------------------------

        df["date"] = pd.to_datetime(df["date"])

        # -------------------------------------------------
        # FILTER MAY -> SEPTEMBER
        # -------------------------------------------------

        df = df[
            (df["date"].dt.month >= START_MONTH) &
            (df["date"].dt.month <= END_MONTH)
        ].copy()

        # -------------------------------------------------
        # CLEAN PRECIPITATION VALUES
        # -------------------------------------------------

        # Convert to string first
        df["rain_mm"] = df["rain_mm"].astype(str)

        # Replace trace precipitation ("T")
        df["rain_mm"] = df["rain_mm"].str.replace("T", "0")

        # Remove spaces
        df["rain_mm"] = df["rain_mm"].str.strip()

        # Convert to numeric
        df["rain_mm"] = pd.to_numeric(
            df["rain_mm"],
            errors="coerce"
        )

        # Remove invalid rows
        df = df.dropna(subset=["rain_mm"])

        # -------------------------------------------------
        # ADD WEEK + YEAR
        # -------------------------------------------------

        df["week"] = df["date"].dt.isocalendar().week

        df["year"] = year

        # -------------------------------------------------
        # DIAGNOSTICS
        # -------------------------------------------------

        seasonal_total = df["rain_mm"].sum()

        print(f"\nSeasonal Rainfall Total: {seasonal_total:.1f} mm")

        if seasonal_total < 100:
            print("WARNING: Seasonal rainfall seems suspiciously low")

        print("\nRainfall Summary:")
        print(df["rain_mm"].describe())

        all_data.append(df)

        # Be polite to government servers
        time.sleep(1)

    except Exception as e:

        print(f"FAILED: {e}")

# =========================================================
# STOP IF NO DATA
# =========================================================

if not all_data:

    print("\nCRITICAL ERROR:")
    print("No weather data downloaded successfully.")

    sys.exit()

# =========================================================
# COMBINE ALL YEARS
# =========================================================

weather_df = pd.concat(all_data, ignore_index=True)

print("\n=================================================")
print("COMBINED DATA SUMMARY")
print("=================================================")

print(weather_df.head())

# =========================================================
# WEEKLY TOTALS
# =========================================================

weekly_totals = (
    weather_df
    .groupby(["year", "week"])["rain_mm"]
    .sum()
    .reset_index()
)

# =========================================================
# CREATE DATE RANGES (UPDATED FOR 2026)
# =========================================================

TARGET_YEAR = 2026

# Get the unique ISO weeks from the dataset
unique_weeks = weather_df["week"].unique()

date_ranges = []

for w in unique_weeks:
    # fromisocalendar takes (year, week, day_of_week) where 1=Monday, 7=Sunday
    start_date = datetime.fromisocalendar(TARGET_YEAR, w, 1)
    end_date = datetime.fromisocalendar(TARGET_YEAR, w, 7)
    
    date_str = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}"
    date_ranges.append({"week": w, "date_range": date_str})

week_ranges = pd.DataFrame(date_ranges)

# =========================================================
# HISTORICAL WEEKLY AVERAGES
# =========================================================

weekly_avg = (
    weekly_totals
    .groupby("week")["rain_mm"]
    .mean()
    .reset_index()
)

weekly_avg.rename(
    columns={
        "rain_mm": "avg_rainfall_mm"
    },
    inplace=True
)

# =========================================================
# SIMPLE PREDICTION
# =========================================================

weekly_avg["predicted_rainfall_mm"] = (
    weekly_avg["avg_rainfall_mm"].round(1)
)

# =========================================================
# MERGE DATE RANGES
# =========================================================

weekly_avg = weekly_avg.merge(
    week_ranges[["week", "date_range"]],
    on="week",
    how="left"
)

# =========================================================
# REORDER COLUMNS
# =========================================================

weekly_avg = weekly_avg[
    [
        "week",
        "date_range",
        "avg_rainfall_mm",
        "predicted_rainfall_mm"
    ]
]

# =========================================================
# ROUND VALUES
# =========================================================

weekly_avg["avg_rainfall_mm"] = (
    weekly_avg["avg_rainfall_mm"].round(1)
)

# =========================================================
# OUTPUT RESULTS
# =========================================================

print("\n=================================================")
print("10-YEAR WEEKLY RAINFALL AVERAGES")
print("=================================================\n")

print(weekly_avg)

# =========================================================
# SAVE CSV
# =========================================================

filename = (
    f"{TARGET_STATION.lower()}_"
    f"weekly_rainfall_averages.csv"
)

weekly_avg.to_csv(
    filename,
    index=False
)

print(f"\nSaved File:")
print(filename)

# =========================================================
# OPTIONAL: SAVE RAW DAILY DATA
# =========================================================

raw_filename = (
    f"{TARGET_STATION.lower()}_raw_daily_weather.csv"
)

weather_df.to_csv(
    raw_filename,
    index=False
)

print(raw_filename)

# =========================================================
# OPTIONAL: SAVE WEEKLY TOTALS
# =========================================================

weekly_totals_filename = (
    f"{TARGET_STATION.lower()}_weekly_totals.csv"
)

weekly_totals.to_csv(
    weekly_totals_filename,
    index=False
)

print(weekly_totals_filename)

print("\nDONE.")