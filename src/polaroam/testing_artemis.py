# %%
import os
os.environ['POLARS_MAX_THREADS'] = '20'
import pandas as pd
import polars as pl
import polaroam
import models
from models import Infostop
import numpy as np
import utils
import time

import glob
from datetime import datetime
import concurrent.futures


# Assuming you have defined `models.Infostop` and it is correctly implemented
# Adjust the directory paths as necessary

# %%
# input_directory = "/data_1/quadrant/output_local/filter_partioned/"
home_directory = os.path.expanduser("~")
input_directory = "data_quadrant/filter_partioned/"
#  output_directory = "/data_1/quadrant/output_local/ollin_stops/"
output_directory = "data_quadrant/ollin_stops/"
columns_to_read = ["_c0", "_c2", "_c3", "_c5"]

# # Get all files in the directory
files = glob.glob(os.path.join(home_directory, input_directory,"day*"), recursive=True)
# files = glob.glob(input_directory, recursive=False)

# # Extract unique dates from the file paths (assuming the date format is within the file path)
# unique_dates = sorted(set([os.path.basename(os.path.dirname(f)).split('=')[-1] for f in files]))
unique_dates = sorted(set([os.path.basename(f).split('=')[-1] for f in files]))

start_date = "2022-12-02"
end_date = "2022-12-08"
filtered_dates = [date for date in unique_dates if date >= start_date]
filtered_dates = [date for date in filtered_dates if date <= end_date]

# By dates
# %%

# for date_str in unique_dates:
def process_date(date_str):
    print(f"Processing date: {date_str}")

    # Load all parquet files for the specific date
    df = pl.scan_parquet(os.path.join(home_directory,input_directory, f"day={date_str}/*.parquet"))

    # Select only necessary columns
    df = df.select(columns_to_read)

    # Rename columns and process timestamps
    df = df.rename({"_c0":"uid", "_c2":"latitude", "_c3":"longitude", "_c5":"timestamp"})
    df = df.with_columns(pl.col("timestamp").cast(pl.Int64))
    df = df.sort("timestamp")
    df = (df.with_columns(
           date = pl.from_epoch("timestamp", time_unit="s")
           )
          .with_columns(
          pl.col("date").dt.replace_time_zone("UTC")
          .dt.replace_time_zone("America/Mexico_City")
          )
          .with_columns(
          pl.col("date").dt.epoch(time_unit="s").alias("timestamp")
          )
        )

    # Initialize your Infostop model
    model = models.Infostop(
        r1=10,
        r2=10,
        min_staying_time=300,
        max_time_between=3600,
        min_size=2,
        distance_metric="haversine",
        verbose=False,
        num_threads=20,
        min_spacial_resolution=0
    )

    # Fit the model and compute stops
    start = time.time()
    labels = model.fit_predict(df)
    medians = model.compute_label_medians()
    stop_locations = model.compute_dbscan()

    # Prepare output file path
    output_file = os.path.join(home_directory, output_directory, f"{date_str}_all_stops.parquet")

    # Save the processed stop locations
    pl.Config.set_streaming_chunk_size(10000)
    stop_locations.collect(streaming=True).write_parquet(output_file, use_pyarrow=True)

    end = time.time()
    print(f"Processed {date_str} in {end - start} seconds")

# %%

# Use ThreadPoolExecutor to process 4 dates at a time
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(process_date, date) for date in filtered_dates]

    # Wait for all futures to complete
    concurrent.futures.wait(futures)

#########################################################
#########################################################
#########################################################
#########################################################
########## Non concurrent
#########################################################
#########################################################
#########################################################
#########################################################

# By dates
# %%

for date_str in filtered_dates:
    print(f"Processing date: {date_str}")

    # Load all parquet files for the specific date
    df = pl.scan_parquet(os.path.join(home_directory,input_directory, f"day={date_str}/*.parquet"))

    # Select only necessary columns
    df = df.select(columns_to_read)

    # Rename columns and process timestamps
    df = df.rename({"_c0":"uid", "_c2":"latitude", "_c3":"longitude", "_c5":"timestamp"})
    df = df.with_columns(pl.col("timestamp").cast(pl.Int64))
    df = df.sort("timestamp")
    df = (df.with_columns(
           date = pl.from_epoch("timestamp", time_unit="s")
           )
          .with_columns(
          pl.col("date").dt.replace_time_zone("UTC")
          .dt.replace_time_zone("America/Mexico_City")
          )
          .with_columns(
          pl.col("date").dt.epoch(time_unit="s").alias("timestamp")
          )
        )

    # Initialize your Infostop model
    model = models.Infostop(
        r1=10,
        r2=10,
        min_staying_time=300,
        max_time_between=3600,
        min_size=2,
        distance_metric="haversine",
        verbose=False,
        num_threads=20,
        min_spacial_resolution=0
    )

    # Fit the model and compute stops
    start = time.time()
    labels = model.fit_predict(df)
    medians = model.compute_label_medians()
    stop_locations = model.compute_dbscan()

    # Prepare output file path
    output_file = os.path.join(home_directory, output_directory, f"{date_str}_all_stops.parquet")

    # Save the processed stop locations
    pl.Config.set_streaming_chunk_size(10000)
    stop_locations.collect(streaming=True).write_parquet(output_file, use_pyarrow=True)

    end = time.time()
    print(f"Processed {date_str} in {end - start} seconds")

#######################################################################################################
#######################################################################################################
#######################################################################################################
########### Re-create the clusters for all the days
#######################################################################################################
#######################################################################################################
#######################################################################################################

output_clusters = "data_quadrant/ollin_clusters/"

df = pl.scan_parquet(os.path.join(home_directory,output_directory, "*.parquet"))
columns_to_read = ["uid", "stop_events", "latitude", "longitude", "start_timestamp", "end_timestamp"]
df = df.select(columns_to_read)
#  df.collect_schema()
#  df = df.rename({"inverse_indices":"index"})
#  df = df.with_row_index("index")
#  df.collect_schema()

model = models.Infostop(
    r1=10,
    r2=10,
    min_staying_time=300,
    max_time_between=3600,
    min_size=2,
    distance_metric="haversine",
    verbose=False,
    num_threads=20,
    min_spacial_resolution=0
)

start = time.time()
model._median_coords = df
stop_locations = model.compute_dbscan()

# Prepare output file path
output_file = os.path.join(home_directory, output_clusters, "all_days_clusters.parquet")

# Save the processed stop locations
pl.Config.set_streaming_chunk_size(10000)
stop_locations.collect(streaming=True).write_parquet(output_file, use_pyarrow=True)

end = time.time()
print(f"Processed the clustering in {end - start} seconds")


# ##########################
# #### Example with quadrant
# # %%
# columns_to_read = ["_c0", "_c2", "_c3", "_c5"]
# # df = pl.scan_parquet("~/data_quadrant/filter_partioned/day=2022-12-04/*.parquet")
# # df = pl.scan_parquet("/data_1/quadrant/output_local/filter_partioned/**/*.parquet")
# df = pl.scan_parquet("~/data_quadrant/filter_partioned/**/*.parquet")

# df = df.select(columns_to_read)

# # %%
# df = df.rename({"_c0":"uid", "_c2":"latitude", "_c3":"longitude", "_c5":"timestamp"})
# df = df.with_columns(pl.col("timestamp")).cast(pl.Int64)
# df = df.sort("timestamp")#.collect()
# df = (df.with_columns(
#        date = pl.from_epoch("timestamp", time_unit="s")
#        )
#       .with_columns(
#       pl.col("date").dt.replace_time_zone("UTC")
#       .dt.replace_time_zone("America/Mexico_City")
#       )
#       .with_columns(
#       pl.col("date").dt.epoch(time_unit="s").alias("timestamp")
#       )
#     )


# # %%
# model = models.Infostop(
#     r1=10,  # Max distance to consider points as stationary
#     r2=10,  # Max distance to consider stationary points as connected
#     min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
#     max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
#     min_size=2,  # Minimum number of points to consider a stop
#     distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
#     verbose=False,  # Set to True for more detailed output during processing
#     num_threads = 20,
#     min_spacial_resolution=0
# )


# # %%
# start = time.time()
# labels = model.fit_predict(df)
# medians = model.compute_label_medians()
# stop_locations = model.compute_dbscan()
# pl.Config.set_streaming_chunk_size(10000)
# (stop_locations
#  .collect(streaming=True)
# #  .write_parquet("/data_1/quadrant/output_local/ollin_stops/stops_whole_period.parquet"
#  .write_parquet("~/data_quadrant/ollin_stops/stops_whole_period.parquet"
#                 , use_pyarrow=True
#                 , compression="snappy"
#                 ))
# end = time.time()
# print(end - start)



# %%
