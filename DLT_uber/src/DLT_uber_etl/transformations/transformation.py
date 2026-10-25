import dlt
from pyspark.sql.functions import (
    col, explode, to_timestamp, when, avg, sum, count, 
    window, expr, size, round, collect_set
)
from pyspark.sql.types import *

SOURCE_PATH = "/Volumes/main/default/raw_data/uber_rides/"

# ==============================================================================
# 1. BRONZE LAYER: Ingest raw streaming Uber events via Auto Loader
# ==============================================================================
@dlt.table(
    name="uber_rides_bronze1",
    comment="Raw streaming Uber ride payloads ingested incrementally via cloudFiles",
    table_properties={"quality": "bronze"}
)
def uber_rides_bronze1():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("multiLine", "true")
        .load(SOURCE_PATH)
        .select(
            "*",
            col("_metadata.file_name").alias("source_file"),
            col("_metadata.file_modification_time").alias("ingestion_time")
        )
    )

# ==============================================================================
# 2. SILVER LAYER: Complex Parsing, Array Explosions, and Data Quality
# ==============================================================================
@dlt.table(
    name="uber_rides_silver",
    comment="Cleaned ride details with flattened structs, exploded waypoints, and validated business rules",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("valid_ride_id", "ride_id IS NOT NULL")
@dlt.expect_or_drop("valid_surge", "surge_multiplier >= 1.0 AND surge_multiplier <= 5.0")
def uber_rides_silver():
    raw_df = dlt.read_stream("uber_rides_bronze1")
    
    flattened = (
        raw_df
        .select(
            col("ride_id"),
            col("city"),
            col("rider.id").alias("rider_id"),
            col("rider.rating").alias("rider_rating"),
            col("driver.id").alias("driver_id"),
            col("driver.name").alias("driver_name"),
            col("driver.vehicle_type").alias("vehicle_type"),
            col("fare_details.base_fare").cast("double").alias("base_fare"),
            col("fare_details.surge_multiplier").cast("double").alias("surge_multiplier"),
            col("fare_details.tips").cast("double").alias("tips"),
            col("applied_promos"),
            explode(col("waypoints")).alias("waypoint"),
            col("ingestion_time")
        )
        .select(
            col("ride_id"),
            col("city"),
            col("rider_id"),
            col("rider_rating"),
            col("driver_id"),
            col("driver_name"),
            col("vehicle_type"),
            col("base_fare"),
            col("surge_multiplier"),
            col("tips"),
            col("applied_promos"),
            col("waypoint.sequence").alias("waypoint_seq"),
            col("waypoint.location").alias("waypoint_location"),
            to_timestamp(col("waypoint.timestamp")).alias("waypoint_timestamp"),
            col("ingestion_time"),
            round(col("base_fare") * col("surge_multiplier") + col("tips"), 2).alias("total_calculated_fare"),
            when(size(col("applied_promos")) > 0, True).otherwise(False).alias("has_promo"),
            when(col("surge_multiplier") >= 2.0, True).otherwise(False).alias("is_high_surge")
        )
    )
    
    return flattened

# ==============================================================================
# 3. GOLD LAYER: Hourly City-Level Aggregations for Executive Dashboard
# ==============================================================================
@dlt.table(
    name="uber_city_performance_gold",
    comment="Hourly windowed metrics aggregated by city and vehicle type for executive reporting",
    table_properties={"quality": "gold"}
)
def uber_city_performance_gold():
    silver_df = dlt.read("uber_rides_silver")
    
    return (
        silver_df
        .groupBy(
            col("city"),
            col("vehicle_type"),
            window(col("waypoint_timestamp"), "1 hour").alias("time_window")
        )
        .agg(
            count("ride_id").alias("total_rides"),
            sum("total_calculated_fare").alias("gross_revenue"),
            avg("surge_multiplier").alias("avg_surge_multiplier"),
            avg("rider_rating").alias("avg_rider_rating"),
            count(when(col("is_high_surge") == True, 1)).alias("high_surge_ride_count"),
            collect_set("driver_id").alias("active_drivers")
        )
        .select(
            col("city"),
            col("vehicle_type"),
            col("time_window.start").alias("window_start"),
            col("time_window.end").alias("window_end"),
            col("total_rides"),
            round(col("gross_revenue"), 2).alias("gross_revenue"),
            round(col("avg_surge_multiplier"), 2).alias("avg_surge_multiplier"),
            round(col("avg_rider_rating"), 2).alias("avg_rider_rating"),
            col("high_surge_ride_count"),
            size(col("active_drivers")).alias("active_driver_count")
        )
    )