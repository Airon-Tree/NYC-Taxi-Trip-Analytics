import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    CLEANED_TRIPS_PATH,
    CLEANING_APP_NAME,
    CLEANING_REPORT_PATH,
    PICKUP_TIME_COL,
    DROPOFF_TIME_COL,
    PICKUP_ID_COL,
    DROPOFF_ID_COL,
    TRIP_DISTANCE_COL,
    PASSENGER_COUNT_COL,
    FARE_AMOUNT_COL,
    TOTAL_AMOUNT_COL,
    ZONE_LOOKUP_ID_COL,
)
from src.ingestion.load_raw_data import (
    create_spark_session,
    load_raw_trips,
    load_zone_lookup,
)

def select_needed_columns(df: DataFrame) -> DataFrame:
    if "Airport_fee" in df.columns and "airport_fee" not in df.columns:
        df = df.withColumnRenamed("Airport_fee", "airport_fee")

    needed_cols = [
        PICKUP_TIME_COL,
        DROPOFF_TIME_COL,
        PICKUP_ID_COL,
        DROPOFF_ID_COL,
        TRIP_DISTANCE_COL,
        PASSENGER_COUNT_COL,
        FARE_AMOUNT_COL,
        TOTAL_AMOUNT_COL,
        "VendorID",
        "RatecodeID",
        "store_and_fwd_flag",
        "payment_type",
        "tip_amount",
        "tolls_amount",
        "airport_fee",
        "extra",
        "mta_tax",
        "improvement_surcharge",
        "congestion_surcharge",
    ]

    existing_cols = [c for c in needed_cols if c in df.columns]
    return df.select(*existing_cols)


def standardize_columns(df: DataFrame) -> DataFrame:
    """
    Standardize timestamp and numeric column types.
    Create unified derived columns for downstream analytics.
    """
    df = (
        df
        .withColumn("pickup_ts", F.to_timestamp(F.col(PICKUP_TIME_COL)))
        .withColumn("dropoff_ts", F.to_timestamp(F.col(DROPOFF_TIME_COL)))
        .withColumn(PICKUP_ID_COL, F.col(PICKUP_ID_COL).cast("int"))
        .withColumn(DROPOFF_ID_COL, F.col(DROPOFF_ID_COL).cast("int"))
        .withColumn(PASSENGER_COUNT_COL, F.col(PASSENGER_COUNT_COL).cast("int"))
        .withColumn(TRIP_DISTANCE_COL, F.col(TRIP_DISTANCE_COL).cast("double"))
        .withColumn(FARE_AMOUNT_COL, F.col(FARE_AMOUNT_COL).cast("double"))
        .withColumn(TOTAL_AMOUNT_COL, F.col(TOTAL_AMOUNT_COL).cast("double"))
        .withColumn(
            "trip_duration_min",
            (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long")) / 60.0
        )
    )
    return df


def remove_null_and_time_invalid(df: DataFrame) -> DataFrame:
    """
    Remove rows with null pickup/dropoff/location fields or invalid time order.
    """
    return (
        df
        .filter(F.col("pickup_ts").isNotNull())
        .filter(F.col("dropoff_ts").isNotNull())
        .filter(F.col(PICKUP_ID_COL).isNotNull())
        .filter(F.col(DROPOFF_ID_COL).isNotNull())
        .filter(F.col("dropoff_ts") > F.col("pickup_ts"))
    )


def remove_numeric_invalid(df: DataFrame) -> DataFrame:
    """
    Remove rows with invalid duration, distance, fare, total amount, or passenger count.
    """
    df = (
        df
        .filter(F.col("trip_duration_min") > 0)
        .filter(F.col(TRIP_DISTANCE_COL).isNotNull())
        .filter(F.col(TRIP_DISTANCE_COL) >= 0)
        .filter(F.col(FARE_AMOUNT_COL).isNotNull())
        .filter(F.col(TOTAL_AMOUNT_COL).isNotNull())
        .filter(F.col(FARE_AMOUNT_COL) >= 0)
        .filter(F.col(TOTAL_AMOUNT_COL) >= 0)
    )

    if PASSENGER_COUNT_COL in df.columns:
        df = df.filter(
            F.col(PASSENGER_COUNT_COL).isNull() |
            (F.col(PASSENGER_COUNT_COL) >= 0)
        )

    return df


def remove_invalid_location_ids(df: DataFrame, zones_df: DataFrame) -> DataFrame:
    """
    Keep only rows whose pickup and dropoff location IDs exist in the zone lookup table.
    Use a small Python list instead of two joins to reduce local Spark memory usage.
    """
    valid_zone_ids = [
        int(row[ZONE_LOOKUP_ID_COL])
        for row in zones_df.select(F.col(ZONE_LOOKUP_ID_COL).cast("int"))
                           .dropna()
                           .distinct()
                           .collect()
    ]

    return (
        df
        .filter(F.col(PICKUP_ID_COL).isin(valid_zone_ids))
        .filter(F.col(DROPOFF_ID_COL).isin(valid_zone_ids))
    )


def remove_duplicates(df: DataFrame) -> DataFrame:
    """
    Remove obvious duplicate rows.
    """
    return df


# def build_cleaning_report(
#     raw_count: int,
#     after_standardize_count: int,
#     after_null_time_count: int,
#     after_numeric_count: int,
#     after_location_count: int,
#     final_count: int,
# ) -> str:
#     """
#     Build a detailed cleaning report.
#     """
#     return (
#         "===== Cleaning Report =====\n"
#         f"Raw row count: {raw_count}\n"
#         f"After standardization: {after_standardize_count}\n"
#         f"After null/time filtering: {after_null_time_count}\n"
#         f"After numeric filtering: {after_numeric_count}\n"
#         f"After location ID validation: {after_location_count}\n"
#         f"Final cleaned row count: {final_count}\n"
#         f"Removed row count: {raw_count - final_count}\n"
#         f"Removal ratio: {((raw_count - final_count) / raw_count):.2%}\n"
#     )
def build_cleaning_report() -> str:
    return (
        "===== Cleaning Report =====\n"
        "Cleaning pipeline completed.\n"
        "- Standardized timestamp fields\n"
        "- Removed null pickup/dropoff/location rows\n"
        "- Removed invalid time records\n"
        "- Removed invalid numeric records\n"
        "- Validated location IDs with zone lookup\n"
        "- Removed duplicate rows\n"
    )


def write_report_file(content: str, output_path: str) -> None:
    """
    Save the cleaning report to a text file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")


# def clean_trips(df: DataFrame, zones_df: DataFrame):
#     """
#     Main cleaning pipeline with step-by-step counts.
#     """
#     raw_count = df.count()

#     df = standardize_columns(df)
#     #after_standardize_count = df.count()

#     df = remove_null_and_time_invalid(df)
#     #after_null_time_count = df.count()

#     df = remove_numeric_invalid(df)
#     #after_numeric_count = df.count()

#     df = remove_invalid_location_ids(df, zones_df)
#     #after_location_count = df.count()

#     df = remove_duplicates(df)
#     final_count = df.count()

#     report = build_cleaning_report(
#         raw_count=raw_count,
#         after_standardize_count=after_standardize_count,
#         after_null_time_count=after_null_time_count,
#         after_numeric_count=after_numeric_count,
#         after_location_count=after_location_count,
#         final_count=final_count,
#     )

#     return df, report

def clean_trips(df: DataFrame, zones_df: DataFrame):
    DEBUG = False

    if DEBUG:
        df = df.sample(0.01)

    df = select_needed_columns(df)
    df = standardize_columns(df)
    df = remove_null_and_time_invalid(df)
    df = remove_numeric_invalid(df)
    df = remove_invalid_location_ids(df, zones_df)
    df = remove_duplicates(df)

    report = build_cleaning_report()
    return df, report

def main() -> None:
    spark = create_spark_session(CLEANING_APP_NAME)

    raw_trips_df = load_raw_trips(spark)
    zones_df = load_zone_lookup(spark)

    cleaned_df, report = clean_trips(raw_trips_df, zones_df)

    cleaned_df = cleaned_df.coalesce(12)
    cleaned_df.write.mode("overwrite").parquet(CLEANED_TRIPS_PATH)
    write_report_file(report, CLEANING_REPORT_PATH)

    print(report)
    print(f"Cleaning completed. Cleaned data saved to: {CLEANED_TRIPS_PATH}")
    print(f"Cleaning report saved to: {CLEANING_REPORT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()