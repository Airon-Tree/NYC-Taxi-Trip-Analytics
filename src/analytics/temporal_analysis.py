import argparse
import sys
from pathlib import Path

from pyspark import StorageLevel
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
    "pickup_service_zone",
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
    "avg_trip_distance",
    "total_trip_distance",
    "avg_trip_duration_min",
    "total_trip_duration_min",
    "avg_fare_amount",
    "total_fare_amount",
    "avg_total_amount",
    "total_revenue",
    "avg_tip_amount",
    "total_tip_amount",
    "total_tolls_amount",
    "total_airport_fee",
    "avg_passenger_count",
    "total_passenger_count",
    "avg_speed_mph",
    "credit_card_trip_count",
    "cash_trip_count",
    "credit_card_share",
    "cash_share",
]


def validate_input_schema(df: DataFrame) -> None:
    """
    Check whether the Stage 3 zone_hour_features table has the columns
    needed by Stage 4.
    """
    missing_cols = [col_name for col_name in REQUIRED_COLUMNS if col_name not in df.columns]

    if missing_cols:
        raise ValueError(
            "Missing required columns in zone_hour_features:\n"
            + "\n".join(f"- {col_name}" for col_name in missing_cols)
            + "\n\nPlease rerun Stage 3: python src/FeatureAndSpatial/zone_hour_features.py"
        )


def add_weighted_metrics(df: DataFrame) -> DataFrame:
    """
    Add weighted average metrics after aggregation.

    Since zone_hour_features is already aggregated, we should not calculate
    plain averages of avg_* columns again. Instead, we recompute averages
    from total columns.
    """
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
            "avg_total_amount",
            F.when(F.col("total_trips") > 0, F.col("total_revenue") / F.col("total_trips")),
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
    """
    Standard aggregation function for most temporal analysis tables.
    """
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


def build_kpi_summary(zone_hour_df: DataFrame) -> DataFrame:
    """
    One-row summary table.
    """
    return (
        zone_hour_df.agg(
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


def build_hourly_demand(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = one hour of day.
    Shows the overall 24-hour taxi demand pattern.
    """
    return aggregate_demand(zone_hour_df, ["hour"]).orderBy("hour")


def build_daily_demand(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = one calendar date.
    Shows daily taxi demand trend.
    """
    return (
        aggregate_demand(
            zone_hour_df,
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


def build_monthly_demand(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = one year-month.
    Shows monthly taxi demand trend.
    """
    return (
        aggregate_demand(
            zone_hour_df,
            [
                "year",
                "month",
                "year_month",
            ],
        )
        .orderBy("year", "month")
    )


def build_weekday_weekend_hourly(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = weekday/weekend + hour.
    Compares weekday and weekend demand patterns.
    """
    return (
        aggregate_demand(zone_hour_df, ["is_weekend", "hour"])
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
            "avg_total_amount",
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


def build_weekday_hour_heatmap(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = weekday + hour.
    This table is ready for a weekday-hour heatmap.
    """
    return (
        aggregate_demand(
            zone_hour_df,
            [
                "day_of_week",
                "weekday_name",
                "hour",
            ],
        )
        .orderBy("day_of_week", "hour")
    )


def build_borough_hourly_pattern(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = pickup borough + hour.
    Compares hourly demand patterns across boroughs.
    """
    return (
        aggregate_demand(
            zone_hour_df,
            [
                "pickup_borough",
                "hour",
            ],
        )
        .orderBy("pickup_borough", "hour")
    )


def build_rush_hour_summary(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = manually defined time period.
    Useful for presentation/report explanation.
    """
    labeled_df = (
        zone_hour_df.withColumn(
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


def build_borough_weekday_summary(zone_hour_df: DataFrame) -> DataFrame:
    """
    One row = borough + weekday.
    Shows how weekly patterns differ by borough.
    """
    return (
        aggregate_demand(
            zone_hour_df,
            [
                "pickup_borough",
                "day_of_week",
                "weekday_name",
            ],
        )
        .orderBy("pickup_borough", "day_of_week")
    )


def build_top_zones_by_hour(zone_hour_df: DataFrame, top_n: int = 10) -> DataFrame:
    """
    For each hour of day, find the top pickup zones by demand.
    """
    zone_hour_totals = aggregate_demand(
        zone_hour_df,
        [
            "hour",
            PICKUP_ID_COL,
            "pickup_zone",
            "pickup_borough",
        ],
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


def build_top_zones_overall(zone_hour_df: DataFrame, top_n: int = 25) -> DataFrame:
    """
    Overall top pickup zones by total trips.
    """
    zone_totals = aggregate_demand(
        zone_hour_df,
        [
            PICKUP_ID_COL,
            "pickup_zone",
            "pickup_borough",
        ],
    )

    return (
        zone_totals.orderBy(F.desc("total_trips"), F.desc("total_revenue"))
        .limit(top_n)
    )


def write_output_table(df: DataFrame, table_name: str, write_csv: bool = False) -> None:
    """
    Save output table as parquet.
    Optionally also save CSV.

    Spark writes parquet/csv as directories containing part files.
    This is normal for Spark jobs.
    """
    PARQUET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PARQUET_OUTPUT_DIR / table_name
    df.write.mode("overwrite").parquet(str(parquet_path))
    print(f"Parquet saved: {parquet_path}")

    if write_csv:
        CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        csv_path = CSV_OUTPUT_DIR / table_name

        # Stage 4 outputs are small aggregated tables, so coalesce(1) is safe here.
        (
            df.coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(str(csv_path))
        )

        print(f"CSV saved: {csv_path}")



def write_stage4_readme(write_csv: bool, top_n: int) -> None:
    """
    Write README explaining Stage 4 outputs.
    This version avoids nested markdown code blocks inside Python strings.
    """
    TEMPORAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_note = (
        "CSV outputs are also generated under outputs/tables/temporal/csv/."
        if write_csv
        else "CSV outputs were not generated. Run with --write-csv if CSV files are needed."
    )

    content = f"""
# Stage 4 - Temporal + Analytics Outputs

# This folder stores the output tables generated by:

# python src/analytics/temporal_analysis.py

## Required Input

# data/processed/zone_hour_features/

# This input is created by Stage 3. Each row represents one pickup zone in one date-hour bucket.

## Output Format

##Main output format:

#outputs/tables/temporal/parquet/

"""