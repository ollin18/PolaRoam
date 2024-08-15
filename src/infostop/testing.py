# %%
import pandas as pd
import polars as pl
import infostop
import models
from models import Infostop
import numpy as np
import utils

# %%
import importlib
importlib.reload(models)
importlib.reload(infostop)
importlib.reload(utils)

# %%
df = pd.read_parquet("../../data/veraset_movement_416.snappy.parquet")

# %%
df = pl.from_pandas(df)
df = df.with_columns([
    (pl.col("datetime").dt.timestamp() / 1000000).round(0).cast(pl.Int64).alias("timestamp")
])
df = df.sort("timestamp")
# df = df.lazy()

# %%
model = Infostop(
    r1=3,  # Max distance to consider points as stationary
    r2=3,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=86400,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30
)

# Use fit_predict to process the data and get stop location labels
#  labels = model.fit_predict(example_data)
# %%
labels = model.fit_predict(df)

# %%
medians = model.compute_label_medians()

# %%
stop_locations = model.compute_infomap()

# %%
u1 = stop_locations.filter(pl.col("uid") == stop_locations["uid"][0])

import pathlib
path: pathlib.Path = "~/Downloads/median_stops_u1_labels.csv"
u1.write_csv(path, separator=",")
path: pathlib.Path = "~/Downloads/median_stops_all_labels.csv"
stop_locations.write_csv(path, separator=",")


# %%
