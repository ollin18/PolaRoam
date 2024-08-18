# %%
import os
# os.environ['POLARS_MAX_THREADS'] = '20'
import pandas as pd
import polars as pl
import infostop
import models
from models import Infostop
import numpy as np
import utils
import time



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

# %%
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

# Use fit_predict to process the data and get stop location labels
#  labels = model.fit_predict(example_data)
# %%
labels = model.fit_predict(df)

# %%
medians = model.compute_label_medians()

# %%
# stop_locations = model.compute_infomap()
stop_locations = model.compute_dbscan()

# %%
stop_locations = stop_locations.collect()

# %%
start = time.time()
labels = model.fit_predict(df)
medians = model.compute_label_medians()
stop_locations = model.compute_dbscan()
stop_locations = stop_locations.collect()
end = time.time()
print(end - start)


# %%
u1 = stop_locations.filter(pl.col("uid") == stop_locations["uid"][0])
u1 = u1.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("cluster_counts"))
    .alias("count")
)
import pathlib
path: pathlib.Path = "~/Downloads/median_stops_u1_labels.csv"
u1.write_csv(path, separator=",")
# path: pathlib.Path = "~/Downloads/median_stops_all_labels.csv"
# stop_locations.write_csv(path, separator=",")

##########################
#### Example with quadrant
# %%
# df = pl.scan_parquet("~/data_quadrant/filter_partioned/day=2022-12-04/*.parquet")
# %%
columns_to_read = ["_c0", "_c2", "_c3", "_c5"]
# df = pl.scan_parquet("~/data_quadrant/filter_partioned/day=2022-12-04/*.parquet")
df = pl.scan_parquet("~/data_quadrant/filter_partioned/**/*.parquet")

df =df.select(columns_to_read)

# %%
df = df.rename({"_c0":"uid", "_c2":"latitude", "_c3":"longitude", "_c5":"timestamp"})
df = df.with_columns(pl.col("timestamp")).cast(pl.Int64)
df = df.sort("timestamp")#.collect()
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


# %%
model = models.Infostop(
    r1=10,  # Max distance to consider points as stationary
    r2=10,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 20,
    min_spacial_resolution=0
)


# %%
start = time.time()
labels = model.fit_predict(df)
medians = model.compute_label_medians()
stop_locations = model.compute_dbscan()
pl.Config.set_streaming_chunk_size(10000)
stop_locations.collect(streaming=True).write_parquet("~/data_quadrant/ollin_stops/stops_whole_period.parquet", use_pyarrow=True)
end = time.time()
print(end - start)






# %%
model2 = models.Infostop(
    r1=3,  # Max distance to consider points as stationary
    r2=3,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=86400,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30
)

# %%
# labels = model.fit_predict(df.collect())
# %%
import time

# %%
start = time.time()
labels = model.fit_predict(df)
end = time.time()
print(end - start)

# %%
start = time.time()
medians = model.compute_label_medians()
end = time.time()
print(end - start)

# %%
model2._median_coords = medians
start = time.time()
stop_locations = model2.compute_dbscan()
end = time.time()
print(end - start)

# %%
labs = model.fit_predict(df2)

# %%
thelabs = labs.collect()