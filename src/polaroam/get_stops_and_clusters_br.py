import polars as pl
from polario.hive_dataset import HiveDataset
import os
import glob
from datetime import datetime

import models
from models import Stopdetect
import numpy as np
import utils
import time
import pathlib

import polaroam

# %%
import importlib
importlib.reload(models)
importlib.reload(polaroam)
importlib.reload(utils)

data_dir = os.path.join("/","data","Berkeley","BR","data_rio")
#  df = pl.scan_parquet(os.path.join(data_dir,"new_filtered_localized","**","*.parquet"))

files = glob.glob(os.path.join(data_dir, "filter_partioned","day*"), recursive=True)

unique_dates = sorted(set([os.path.basename(f).split('=')[-1] for f in files]))
date_str='2023-11-19'

for date_str in unique_dates:
    print(f"Processing date: {date_str}")

    # Load all parquet files for the specific date
    df = pl.scan_parquet(os.path.join(data_dir,"filter_partioned", f"day={date_str}/*.parquet"))
    df = df.with_columns(pl.col("id").alias("uid"),
                    pl.col("lat").alias("latitude"),
                    pl.col("lon").alias("longitude"),
                    pl.col("ts").alias("timestamp")
                    )
    df.collect_schema()
    #  df = df.filter(pl.col("error") <= 20)

    # Select only necessary columns
    #  df = df.select(columns_to_read)

    # Initialize your Infostop model

    model = models.Stopdetect(
        r1=20,  # Max distance to consider points as stationary
        r2=20,  # Max distance to consider stationary points as connected
        min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
        max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
        min_size=2,  # Minimum number of points to consider a stop
        distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
        verbose=False,  # Set to True for more detailed output during processing
        num_threads = 30,
        min_spacial_resolution=0
    )

    # Fit the model and compute stops
    start = time.time()
    labels = model.fit_predict(df)
    medians = model.compute_label_medians()

    # Prepare output file path
    #  ds = HiveDataset(os.path.join(data_dir,f"stops_r1{model.r1}_ms{model.min_staying_time}_mt{md.max_time_between}","all_users_stops"), partition_columns=["date"])

    pl.Config.set_streaming_chunk_size(10000)

    #  ds.write(medians.collect(streaming=True))
    base_path = os.path.join(data_dir,f"stops_r1{model._r1}_ms{model._min_staying_time}_mt{model._max_time_between}","all_users_stops")
    os.makedirs(base_path, exist_ok=True)
    output_file = os.path.join(base_path, f"{date_str}.parquet")
    medians.collect(streaming=True).write_parquet(output_file, use_pyarrow=True)

    end = time.time()
    print(f"Processed {date_str} in {end - start} seconds")


#  df = pl.scan_parquet(os.path.join(data_dir,f"stops_r1{model.r1}_ms{model.min_staying_time}_mt{md.max_time_between}","all_users_stops",**,"*.parquet"))
df = pl.scan_parquet(os.path.join(base_path,"*.parquet"))
columns_to_read = ["uid", "stop_events", "latitude", "longitude", "start_timestamp", "end_timestamp"]
df = df.select(columns_to_read)

model = models.Stopdetect(
    r1=20,  # Max distance to consider points as stationary
    r2=20,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30,
    min_spacial_resolution=0.01
)

start = time.time()
model._median_coords = df
stop_locations = model.compute_dbscan()

# Prepare output file path
#  output_file = os.path.join(data_dir,f"stops_r1{model._r1}_ms{model._min_staying_time}_mt{model._max_time_between}","whole_period_clusters.parquet")
output_file = os.path.join(base_path,"whole_period_clusters.parquet")
# Save the processed stop locations
pl.Config.set_streaming_chunk_size(10000)
stop_locations.collect(streaming=True).write_parquet(output_file, use_pyarrow=True)

end = time.time()
print(f"Processed the clustering in {end - start} seconds")

###########################
### Test if we need to compute the total_days
# %%
hw_model = models.HWEstimate(
    r1=20,  # Max distance to consider points as stationary
    r2=20,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30,
    min_spacial_resolution=0,
    start_hour_day=7,
    end_hour_day=21,
    start_working_hour=8,
    end_working_hour=18,
    min_periods_over_window_home=0.08,
    span_period_home=0.08,
    min_periods_over_window_work=0.05,
    span_period_work=0.05,
    total_days=31,
    convert_tz=False,
    tz="America/Sao_Paulo"  # Or another timezone if different from default
)

data_dir = os.path.join("/","data","Berkeley","BR","data_rio")
base_path = os.path.join(data_dir,f"stops_r1{hw_model._r1}_ms{hw_model._min_staying_time}_mt{hw_model._max_time_between}")
input_file = os.path.join(base_path,"whole_period_clusters.parquet")
stop_locations = pl.scan_parquet(input_file)

hw_df = hw_model.prepare_labeling(stop_locations)
hw_df = hw_model.detect_home()
hw_df = hw_model.detect_work()
hw_df = hw_df.with_columns([
    pl.col("date").dt.date().alias("date_trunc").cast(pl.String)
])

hw_df.collect_schema()

start = time.time()

ds = HiveDataset(os.path.join(base_path,"stops_hw"), partition_columns=["date_trunc"])

pl.Config.set_streaming_chunk_size(10000)
ds.write(hw_df.collect(streaming=True))

end = time.time()
print(end - start)



us1 = pl.read_parquet(os.path.join(base_path, "stops_hw", "**", "*.parquet"))
us1["uid"].unique().len()
us1.filter(pl.col("location_type") == "H")["uid"].unique().len()
us1.filter(pl.col("location_type") == "W")["uid"].unique().len()
us1.collect_schema()


only_hw = us1.select("uid", "location_type", "cluster_latitude", "cluster_longitude")
only_hw = only_hw.filter(pl.col("location_type") != "O").unique(subset=["uid","location_type"])

path: pathlib.Path = os.path.join(base_path,"home_work_locations_by_counts.csv")
only_hw.write_csv(path)


wide_df = only_hw.pivot(
    values=["cluster_latitude", "cluster_longitude"],  # Values to pivot
    index="uid",            # Index to group by
    columns="location_type",     # Column to pivot
    aggregate_function=None # No aggregation needed
).select(
    "uid",
    pl.col("cluster_latitude_H").alias("h_lat"),  # Rename columns for clarity
    pl.col("cluster_longitude_H").alias("h_lon"),
    pl.col("cluster_latitude_W").alias("w_lat"),
    pl.col("cluster_longitude_W").alias("w_lon")
)

path: pathlib.Path = os.path.join(base_path,"hw_arcs_by_counts.csv")
wide_df.write_csv(path)
