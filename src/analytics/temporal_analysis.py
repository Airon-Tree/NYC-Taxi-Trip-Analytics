import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    ANALYTICS_APP_NAME,
    ZONE_HOUR_FEATURES_PATH,
    TABLES_DIR,
    PICKUP_ID_COL,
)

from src.ingestion.load_raw_data import create_spark_session


TEMPORAL_OUTPUT_DIR = TABLES_DIR / "temporal"
PARQUET_OUTPUT_DIR = TEMPORAL_OUTPUT_DIR / "parquet"
CSV_OUTPUT_DIR = TEMPORAL_OUTPUT_DIR / "csv"
README_OUTPUT_PATH = TEMPORAL_OUTPUT_DIR / "README.md"


REQUIRED_COLUMNS = [
    PICKUP_ID_COL,
    "pickup_zone",
    "pickup_borough",
    "pickup_date",
    "year",
    "month",
    "day",
    "year_month",
    "day_of_week",
    "weekday_name",
    "is_weekend",
    "hour",
    "trip_count",
    "total_revenue",
    "total_fare_amount",
    "total_trip_distance",
    "total_trip_duration_min",
    "total_passenger_count",
    "credit_card_trip_count",
    "cash_trip_count",
]


def validate_input_schema(df: DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_cols:
        raise ValueError(
            "Missing columns in zone_hour_features: "
            + ", ".join(missing_cols)
            + "\nPlease rerun Stage 3 first."
        )


def add_weighted_metrics(df: DataFrame) -> DataFrame:
    return (
        df.withColumn(
            "avg_revenue_per_trip",
            F.when(F.col("total_trips") > 0, F.col("total_revenue") / F.col("total_trips")),
        )
        .withColumn(
            "avg_fare_amount",
            F.when(F.col("total_trips") > 0, F.col("total_fare_amount") / F.col("total_trips")),
        )
        .withColumn(
            "avg_trip_distance",
            F.when(F.col("total_trips") > 0, F.col("total_trip_distance") / F.col("total_trips")),
        )
        .withColumn(
            "avg_trip_duration_min",
            F.when(
                F.col("total_trips") > 0,
                F.col("total_trip_duration_min") / F.col("total_trips"),
            ),
        )
        .withColumn(
            "avg_passenger_count",
            F.when(
                F.col("total_trips") > 0,
                F.col("total_passenger_count") / F.col("total_trips"),
            ),
        )
        .withColumn(
            "credit_card_share",
            F.when(
                F.col("total_trips") > 0,
                F.col("credit_card_trip_count") / F.col("total_trips"),
            ),
        )
        .withColumn(
            "cash_share",
            F.when(
                F.col("total_trips") > 0,
                F.col("cash_trip_count") / F.col("total_trips"),
            ),
        )
    )


def aggregate_demand(df: DataFrame, group_cols: list[str]) -> DataFrame:
    agg_df = (
        df.groupBy(*group_cols)
        .agg(
            F.sum("trip_count").alias("total_trips"),
            F.sum("total_revenue").alias("total_revenue"),
            F.sum("total_fare_amount").alias("total_fare_amount"),
            F.sum("total_trip_distance").alias("total_trip_distance"),
            F.sum("total_trip_duration_min").alias("total_trip_duration_min"),
            F.sum("total_passenger_count").alias("total_passenger_count"),
            F.sum("credit_card_trip_count").alias("credit_card_trip_count"),
            F.sum("cash_trip_count").alias("cash_trip_count"),
            F.countDistinct("pickup_date").alias("active_days"),
            F.countDistinct(PICKUP_ID_COL).alias("active_pickup_zones"),
        )
    )

    return add_weighted_metrics(agg_df)


def build_kpi_summary(df: DataFrame) -> DataFrame:
    return (
        df.agg(
            F.sum("trip_count").alias("total_trips"),
            F.sum("total_revenue").alias("total_revenue"),
            F.min("pickup_date").alias("start_date"),
            F.max("pickup_date").alias("end_date"),
            F.countDistinct("pickup_date").alias("active_days"),
            F.countDistinct(PICKUP_ID_COL).alias("active_pickup_zones"),
            F.countDistinct("pickup_borough").alias("active_boroughs"),
        )
        .withColumn(
            "avg_trips_per_day",
            F.when(F.col("active_days") > 0, F.col("total_trips") / F.col("active_days")),
        )
        .withColumn(
            "avg_revenue_per_day",
            F.when(F.col("active_days") > 0, F.col("total_revenue") / F.col("active_days")),
        )
    )


def build_hourly_demand(df: DataFrame) -> DataFrame:
    return aggregate_demand(df, ["hour"]).orderBy("hour")


def build_daily_demand(df: DataFrame) -> DataFrame:
    return (
        aggregate_demand(
            df,
            [
                "pickup_date",
                "year",
                "month",
                "day",
                "day_of_week",
                "weekday_name",
                "is_weekend",
            ],
        )
        .orderBy("pickup_date")
    )


def build_monthly_demand(df: DataFrame) -> DataFrame:
    return aggregate_demand(df, ["year", "month", "year_month"]).orderBy("year", "month")


def build_weekday_weekend_hourly(df: DataFrame) -> DataFrame:
    return (
        aggregate_demand(df, ["is_weekend", "hour"])
        .withColumn(
            "day_type",
            F.when(F.col("is_weekend") == 1, F.lit("weekend")).otherwise(F.lit("weekday")),
        )
        .select(
            "day_type",
            "is_weekend",
            "hour",
            "total_trips",
            "total_revenue",
            "avg_revenue_per_trip",
            "avg_fare_amount",
            "avg_trip_distance",
            "avg_trip_duration_min",
            "avg_passenger_count",
            "credit_card_share",
            "cash_share",
            "active_days",
            "active_pickup_zones",
        )
        .orderBy("is_weekend", "hour")
    )


def build_weekday_hour_heatmap(df: DataFrame) -> DataFrame:
    return (
        aggregate_demand(df, ["day_of_week", "weekday_name", "hour"])
        .orderBy("day_of_week", "hour")
    )


def build_borough_hourly_pattern(df: DataFrame) -> DataFrame:
    return (
        aggregate_demand(df, ["pickup_borough", "hour"])
        .orderBy("pickup_borough", "hour")
    )


def build_rush_hour_summary(df: DataFrame) -> DataFrame:
    labeled_df = (
        df.withColumn(
            "time_period_order",
            F.when((F.col("hour") >= 0) & (F.col("hour") <= 5), F.lit(1))
            .when((F.col("hour") >= 6) & (F.col("hour") <= 10), F.lit(2))
            .when((F.col("hour") >= 11) & (F.col("hour") <= 15), F.lit(3))
            .when((F.col("hour") >= 16) & (F.col("hour") <= 19), F.lit(4))
            .otherwise(F.lit(5)),
        )
        .withColumn(
            "time_period",
            F.when((F.col("hour") >= 0) & (F.col("hour") <= 5), F.lit("late_night_00_05"))
            .when((F.col("hour") >= 6) & (F.col("hour") <= 10), F.lit("morning_peak_06_10"))
            .when((F.col("hour") >= 11) & (F.col("hour") <= 15), F.lit("midday_11_15"))
            .when((F.col("hour") >= 16) & (F.col("hour") <= 19), F.lit("evening_peak_16_19"))
            .otherwise(F.lit("night_20_23")),
        )
    )

    return (
        aggregate_demand(labeled_df, ["time_period_order", "time_period"])
        .orderBy("time_period_order")
    )


def build_top_zones_by_hour(df: DataFrame, top_n: int) -> DataFrame:
    zone_hour_totals = aggregate_demand(
        df,
        ["hour", PICKUP_ID_COL, "pickup_zone", "pickup_borough"],
    )

    rank_window = Window.partitionBy("hour").orderBy(
        F.desc("total_trips"),
        F.desc("total_revenue"),
    )

    return (
        zone_hour_totals.withColumn("zone_rank_in_hour", F.dense_rank().over(rank_window))
        .filter(F.col("zone_rank_in_hour") <= top_n)
        .orderBy("hour", "zone_rank_in_hour")
    )


def build_top_zones_overall(df: DataFrame, top_n: int = 25) -> DataFrame:
    zone_totals = aggregate_demand(
        df,
        [PICKUP_ID_COL, "pickup_zone", "pickup_borough"],
    )

    return zone_totals.orderBy(F.desc("total_trips"), F.desc("total_revenue")).limit(top_n)


def write_output_table(df: DataFrame, table_name: str, write_csv: bool) -> None:
    PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PARQUET_OUTPUT_DIR / table_name
    df.write.mode("overwrite").parquet(str(parquet_path))
    print(f"Parquet saved: {parquet_path}")

    if write_csv:
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = CSV_OUTPUT_DIR / table_name
        (
            df.coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(str(csv_path))
        )
        print(f"CSV saved: {csv_path}")


def write_stage4_readme(write_csv: bool, top_n: int) -> None:
    TEMPORAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_note = (
        "CSV outputs are also generated under outputs/tables/temporal/csv/."
        if write_csv
        else "CSV outputs were not generated. Run with --write-csv if CSV files are needed."
    )

    content = f"""
# Stage 4 - Temporal + Analytics Outputs

This folder stores Stage 4 temporal analytics output tables.

Run command:

python src/analytics/temporal_analysis.py

Required input:

data/processed/zone_hour_features/

Main output:

outputs/tables/temporal/parquet/

{csv_note}

Demand definition:

pickup demand = trip_count

Output tables:

1. kpi_summary
- One-row summary of total trips, revenue, date range, active days, and active zones.

2. hourly_demand
- One row per hour of day.
- Used to analyze 24-hour taxi demand pattern.

3. daily_demand
- One row per pickup date.
- Used to analyze daily demand trend.

4. monthly_demand
- One row per year-month.
- Used to analyze monthly demand trend.

5. weekday_weekend_hourly
- One row per weekday/weekend flag and hour.
- Used to compare weekday and weekend hourly patterns.

6. weekday_hour_heatmap
- One row per weekday and hour.
- Ready for weekday-hour heatmap visualization.

7. borough_hourly_pattern
- One row per pickup borough and hour.
- Used to compare hourly demand across boroughs.

8. rush_hour_summary
- One row per time period.
- Time periods: late_night_00_05, morning_peak_06_10, midday_11_15, evening_peak_16_19, night_20_23.

9. top_zones_by_hour
- Top {top_n} pickup zones for each hour.

10. top_zones_overall
- Overall top pickup zones by total trips.

Notes:
- Stage 4 does not read raw trip records directly.
- Stage 4 reads the Stage 3 zone_hour_features table.
- Spark writes parquet and CSV outputs as folders containing part files.
"""

    README_OUTPUT_PATH.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"README saved: {README_OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 - Temporal + Analytics")
    parser.add_argument(
        "--input",
        default=ZONE_HOUR_FEATURES_PATH,
        help="Input path for Stage 3 zone_hour_features parquet.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top pickup zones to keep for each hour.",
    )

    args = parser.parse_args()

    spark = create_spark_session(ANALYTICS_APP_NAME)

    zone_hour_df = spark.read.parquet(args.input)
    validate_input_schema(zone_hour_df)

    print("\n===== Stage 4 input schema =====")
    zone_hour_df.printSchema()

    print("\n===== Stage 4 input preview =====")
    zone_hour_df.show(10, truncate=False)

    output_tables = {
        "kpi_summary": build_kpi_summary(zone_hour_df),
        "hourly_demand": build_hourly_demand(zone_hour_df),
        "daily_demand": build_daily_demand(zone_hour_df),
        "monthly_demand": build_monthly_demand(zone_hour_df),
        "weekday_weekend_hourly": build_weekday_weekend_hourly(zone_hour_df),
        "weekday_hour_heatmap": build_weekday_hour_heatmap(zone_hour_df),
        "borough_hourly_pattern": build_borough_hourly_pattern(zone_hour_df),
        "rush_hour_summary": build_rush_hour_summary(zone_hour_df),
        "top_zones_by_hour": build_top_zones_by_hour(zone_hour_df, args.top_n),
        "top_zones_overall": build_top_zones_overall(zone_hour_df),
    }

    for table_name, table_df in output_tables.items():
        print(f"\n===== Writing {table_name} =====")
        table_df.show(20, truncate=False)
        write_output_table(table_df, table_name, args.write_csv)

    write_stage4_readme(args.write_csv, args.top_n)

    spark.stop()
    print("\nStage 4 Temporal + Analytics completed successfully.")


if __name__ == "__main__":
    main()