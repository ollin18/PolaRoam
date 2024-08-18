# %%
import os
os.environ['POLARS_MAX_THREADS'] = '20'
import pandas as pd
import polars as pl
import infostop
import models
from models import Infostop
import numpy as np
import utils
import time
import tracemalloc



# %%
import importlib
importlib.reload(models)
importlib.reload(infostop)
importlib.reload(utils)


# %%
df = pd.read_parquet("../../data/veraset_movement_416.snappy.parquet")
# df2 = pl.read_parquet("../../data/veraset_movement_416.snappy.parquet")

# %%
df = pl.from_pandas(df)
df = df.lazy()
df = df.with_columns([
    (pl.col("datetime").dt.timestamp() / 1000000).round(0).cast(pl.Int64).alias("timestamp")
])
df = df.sort("timestamp")
# df = df.lazy()


def haversine_polars(lat1, lon1):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = pl.col(lat1).radians()
    lon1_rad = pl.col(lon1).radians()
    lat2_rad = pl.col(lat1).shift(-1).radians()
    lon2_rad = pl.col(lon1).shift(-1).radians()

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
            ((dlat / 2).sin()).pow(2) +
            lat1_rad.cos() * lat2_rad.cos() * ((dlon / 2).sin()).pow(2)

         )

    c = 2 * a.sqrt().arcsin()

    # Radius of Earth in meters
    R = 6371000
    distance = R * c

    return distance

# Example usage with Polars DataFrame
def calculate_distances(coords):
    coords = coords.with_columns([
        haversine_polars(
            'latitude', 'longitude'
        ).alias('distance'),
        (pl.col('timestamp').shift(-1) - pl.col('timestamp')).alias('time_diff')
    ])
    return coords


processed_coords = calculate_distances(df)
#  processed_coords.collect()  # To see the final result
#  df.collect()


def haversine(lat1, lon1, lat2, lon2):
    # Ensure input arrays are numpy arrays for vectorized operations
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])

    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in meters
    R = 6371000
    distance = R * c

    return distance

old_v = df.with_columns([
        haversine(
            pl.col('latitude').shift(-1), pl.col('longitude').shift(-1),
            pl.col('latitude'), pl.col('longitude')
        ).alias('distance'),
        (pl.col('timestamp').shift(-1) - pl.col('timestamp')).alias('time_diff')
    ])


start = time.time()
old_v.collect()
end = time.time()
print(f"Processed old version in {end - start} seconds")

start = time.time()
processed_coords.collect()  # To see the final result
end = time.time()
print(f"Processed polars version in {end - start} seconds")




# starting the monitoring
tracemalloc.start()
old_v.collect()
print(tracemalloc.get_traced_memory())
tracemalloc.stop()

tracemalloc.start()
processed_coords.collect()
print(tracemalloc.get_traced_memory())
tracemalloc.stop()

coords = processed_coords

# Create a mask for stationary events based on thresholds
r_C = 10
max_staying_time = 3600
min_size = 2
min_staying_time = 300

def get_stationary_events_polars(input_df, r_C, min_size, min_staying_time, max_staying_time, distance_metric):
    coords = input_df.select(['uid', 'latitude', 'longitude', 'timestamp'])

    coords = calculate_distances(coords)

    coords = coords.with_columns([
        (pl.col('distance') <= r_C).alias('within_radius'),
        (pl.col('time_diff').is_null() | (pl.col('time_diff') <= max_staying_time)).alias('within_time')
    ])

    # Find clusters of points that meet the stationary criteria
    coords = coords.with_columns([
    (pl.col('within_radius') & pl.col('within_time')).alias('stationary')
    ])

    coords = coords.with_columns([
        (pl.col('stationary') & (~pl.col('stationary').shift(1, fill_value=False))).cast(pl.Int32).alias('event_change')
    ])

    # Create event IDs by cumulatively summing up the transitions
    coords = coords.with_columns([
        (pl.col('event_change').cum_sum().alias('event_id'))
    ])

    # Filter events based on min_size and min_staying_time
    event_stats = coords.group_by('event_id').agg([
        pl.col('event_id').count().alias('event_size'),
        pl.col('time_diff').sum().alias('total_time')
    ]).filter(
        (pl.col('event_size') >= min_size) & (pl.col('total_time') >= min_staying_time)
    )

    # Join coords with event_stats to filter valid event_ids
    coords = coords.join(
        event_stats.select("event_id"),
        on="event_id",
        how="left"
    )

    # Now replace null event_ids with -1
    coords = coords.with_columns(
        pl.col("event_id").fill_null(-1)
    )

    coords = coords.with_columns(
        pl.concat_list(["latitude", "longitude"]).list.to_array(2).alias("stat_coords")
    )

    out = coords.select(
        pl.col("uid"),
        pl.col("event_id").cast(pl.Int64).alias("stop_events"),
        pl.col("stat_coords").alias("event_maps"),
        pl.col('timestamp')
    )

    return out



r_C = 10
min_size = 2
max_staying_time = 3600
min_staying_time = 300
distance_metric = "haversine"

out_polars = get_stationary_events_polars(df, r_C, min_size, min_staying_time, max_staying_time, distance_metric)


tracemalloc.start()
start = time.time()

grouped_data = df.group_by('uid', maintain_order=True)

output_schema = {
    "uid": pl.String,
    "stop_events": pl.Int64,
    "event_maps": pl.Array(pl.Float64, shape=(2,)),
    "timestamp": pl.Int64,
}
# Use apply to handle the stationary events per group in parallel
results = grouped_data.map_groups(
    lambda df: pl.DataFrame(
        get_stationary_events_polars(
            df.select(['uid', 'latitude', 'longitude', 'timestamp']),
            r_C,
            min_size,
            min_staying_time,
            max_staying_time,
            distance_metric
        )
    )
    , schema = output_schema
)
results.collect()

#  data = get_stationary_events_polars(df, r_C, min_size, min_staying_time, max_staying_time, distance_metric)
#  data = out_polars.collect()

end = time.time()
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
print(f"Processed polars version in {end - start} seconds")


model = models.Infostop(
    r1=10,  # Max distance to consider points as stationary
    r2=10,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30,
    min_spacial_resolution=0
)


tracemalloc.start()
start = time.time()

labels = model.fit_predict(df)
labels.collect()

end = time.time()
print(tracemalloc.get_traced_memory())
tracemalloc.stop()
print(f"Processed old version in {end - start} seconds")


