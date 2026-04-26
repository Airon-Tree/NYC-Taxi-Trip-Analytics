import sys
from io import StringIO
from pathlib import Path
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config.config import (
    RAW_DATA_DIR,
    LOOKUP_PATH,
    INGESTION_APP_NAME,
    INGESTION_SUMMARY_PATH,
    PASSENGER_COUNT_COL,
    TRIP_DISTANCE_COL,
    FARE_AMOUNT_COL,
    TOTAL_AMOUNT_COL,
    PICKUP_ID_COL,
    DROPOFF_ID_COL,
)


# def create_spark_session(app_name: str = INGESTION_APP_NAME) -> SparkSession:
#     """
#     Create and return a SparkSession for the ingestion step.
#     """
#     return SparkSession.builder.appName(app_name).getOrCreate()

# def create_spark_session(app_name: str = INGESTION_APP_NAME) -> SparkSession:
#     """
#     Create and return a SparkSession for the ingestion step.
#     """
#     return SparkSession.builder.appName(app_name).getOrCreate()

def create_spark_session(app_name=INGESTION_APP_NAME):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.ip", "127.0.0.1")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "12")
        .config("spark.default.parallelism", "12")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark

def get_parquet_file_paths(raw_data_dir: Path) -> list[str]:
    """
    Find all parquet files under data/raw/.

    We use Python's glob instead of passing '*.parquet' directly into Spark.
    This avoids the noisy metadata warning seen with wildcard paths.
    """
    file_paths = sorted(str(p) for p in raw_data_dir.glob("*.parquet"))
    if not file_paths:
        raise FileNotFoundError(f"No parquet files found in: {raw_data_dir}")
    return file_paths


def normalize_trip_schema(df: DataFrame) -> DataFrame:
    """
    Normalize important columns to a canonical schema.

    This is necessary because different monthly TLC parquet files may store
    some columns with slightly different types. For example, passenger_count
    may be INT64 in one file and be inferred differently in another.
    """
    cast_map = {
        PASSENGER_COUNT_COL: "double",
        TRIP_DISTANCE_COL: "double",
        FARE_AMOUNT_COL: "double",
        TOTAL_AMOUNT_COL: "double",
        PICKUP_ID_COL: "int",
        DROPOFF_ID_COL: "int",
    }

    for column_name, target_type in cast_map.items():
        if column_name in df.columns:
            df = df.withColumn(column_name, col(column_name).cast(target_type))

    return df


def load_raw_trips(spark: SparkSession, raw_data_dir: Path = RAW_DATA_DIR) -> DataFrame:
    """
    Load all raw Yellow Taxi parquet files one by one, normalize their schema,
    and union them into a single Spark DataFrame.

    Reading files individually is more robust than reading all parquet files
    at once when monthly schemas are not perfectly identical.
    """
    file_paths = get_parquet_file_paths(raw_data_dir)

    dataframes = []
    for file_path in file_paths:
        df = spark.read.parquet(file_path)
        df = normalize_trip_schema(df)
        dataframes.append(df)

    combined_df = reduce(lambda left, right: left.unionByName(right, allowMissingColumns=True), dataframes)
    return combined_df


def load_zone_lookup(spark: SparkSession, lookup_path: str = LOOKUP_PATH) -> DataFrame:
    """
    Load the taxi zone lookup CSV into a Spark DataFrame.
    """
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(lookup_path)
    )


def dataframe_to_schema_string(df: DataFrame) -> str:
    """
    Capture df.printSchema() output as a string.
    """
    buffer = StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = buffer
        df.printSchema()
    finally:
        sys.stdout = original_stdout
    return buffer.getvalue()


def dataframe_to_sample_string(df: DataFrame, n: int = 5) -> str:
    """
    Convert top n rows to a string for saving in the summary file.
    """
    rows = df.limit(n).collect()
    if not rows:
        return "[No rows available]"
    return "\n".join(str(row) for row in rows)


def build_dataframe_summary(df: DataFrame, name: str) -> str:
    """
    Build a text summary for a DataFrame:
    - schema
    - row count
    - sample rows
    """
    schema_str = dataframe_to_schema_string(df)
    row_count = df.count()
    sample_str = dataframe_to_sample_string(df, n=5)

    return (
        f"===== {name} Schema =====\n"
        f"{schema_str}\n"
        f"===== {name} Row Count =====\n"
        f"{row_count}\n\n"
        f"===== {name} Sample Rows =====\n"
        f"{sample_str}\n\n"
    )


def write_summary_file(content: str, output_path: str) -> None:
    """
    Write the ingestion summary text to a file under data/processed/.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")


def main() -> None:
    """
    Main workflow:
    1. Create Spark session
    2. Load and normalize raw trip data
    3. Load zone lookup table
    4. Build summary text
    5. Save summary text to processed/
    6. Print only one final completion message
    """
    spark = create_spark_session()

    raw_trips_df = load_raw_trips(spark)
    zones_df = load_zone_lookup(spark)

    summary_parts = [
        build_dataframe_summary(raw_trips_df, "Raw Trips"),
        build_dataframe_summary(zones_df, "Zone Lookup"),
    ]
    final_summary = "\n".join(summary_parts)

    write_summary_file(final_summary, INGESTION_SUMMARY_PATH)

    spark.stop()

    print(f"Ingestion completed. Summary saved to: {INGESTION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()