# %%
import os
# os.environ['POLARS_MAX_THREADS'] = '20'
import pandas as pd
import polars as pl
import polaroam
import models
from models import Stopdetect
import numpy as np
import utils
import time

# %%
import importlib
importlib.reload(models)
importlib.reload(polaroam)
importlib.reload(utils)

# %%
org = pd.read_parquet("../../data/veraset_movement_416.snappy.parquet")
# df2 = pl.read_parquet("../../data/veraset_movement_416.snappy.parquet")

# %%
df = pl.from_pandas(org)
df = df.lazy()
df = df.with_columns([
    (pl.col("datetime").dt.timestamp() / 1000000).round(0).cast(pl.Int64).alias("timestamp")
])
df = df.sort("timestamp")

# %%
hw_model = models.HWEstimate(
    r1=5,  # Max distance to consider points as stationary
    r2=10,  # Max distance to consider stationary points as connected
    min_staying_time=900,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30,
    min_spacial_resolution=0,
    start_hour_day=7,
    end_hour_day=21,
    min_periods_over_window=0.5,
    span_period=0.5,
    total_days=730,
    tz="America/Mexico_City"  # Or another timezone if different from default
)

# %%
labels = hw_model.fit_predict(df)
medians = hw_model.compute_label_medians()
stop_locations = hw_model.compute_dbscan()


# %%
hw_df = hw_model.prepare_labeling(stop_locations)
hw_df = hw_model.detect_home()
hw_df = hw_model.detect_work()

# %%
start = time.time()
hw = hw_df.collect()
end = time.time()
print(end - start)

# %%
# hw
# %%

###########################
### Test if we need to compute the total_days
# %%
hw_model = models.HWEstimate(
    r1=5,  # Max distance to consider points as stationary
    r2=10,  # Max distance to consider stationary points as connected
    min_staying_time=900,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30,
    min_spacial_resolution=0,
    start_hour_day=7,
    end_hour_day=21,
    min_periods_over_window=0.5,
    span_period=0.5,
    total_days=None,
    tz="America/Mexico_City"  # Or another timezone if different from default
)


labels = hw_model.fit_predict(df)
medians = hw_model.compute_label_medians()
stop_locations = hw_model.compute_dbscan()



hw_df = hw_model.prepare_labeling(stop_locations)
hw_df = hw_model.detect_home()
hw_df = hw_model.detect_work()


start = time.time()
hw = hw_df.collect()
end = time.time()
print(end - start)








# %%
