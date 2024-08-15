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
rem = labels.filter(pl.col("stop_events") != -1)
rem = rem.with_columns(pl.col("event_maps").arr.get(0).alias("latitude"),
                 pl.col("event_maps").arr.get(1).alias("longitude"))

# %%
rem = rem.group_by(["uid", "stop_events"], maintain_order=True).agg(pl.col("latitude").median(), pl.col("longitude").median())

# %%

u1 = rem.filter(pl.col("uid") == rem["uid"][0])

import pathlib
path: pathlib.Path = "~/Downloads/median_stops_u1.csv"
u1.write_csv(path, separator=",")

# r2=1
# min_staying_time=300
# max_time_between=86400
# min_size=2
# distance_metric="haversine"
# verbose=True  






# %%
# Pandas
u1 = df.loc[df.uid == df.iloc[0].uid]
u1 = u1.assign(unix_time=u1['datetime'].apply(lambda x: x.timestamp()))

data = u1[["latitude", "longitude", "unix_time"]]
data = data.sort_values(by="unix_time")


# Polars
# %%
df = pl.from_pandas(df)
df = df.with_columns([
    (pl.col("datetime").dt.timestamp() / 1000000).round(0).cast(pl.Int64).alias("timestamp")
])
df = df.sort("timestamp")

# %%
first_uid = df.select(pl.col("uid")).to_numpy()[0][0]
u1 = df.filter(pl.col('uid') == first_uid)

# Select specific columns
data = u1.select(["latitude", "longitude", "timestamp"])

# Sort the DataFrame by "unix_time"
data = data.sort("timestamp")

#  example_data = data.to_numpy()

# Instantiate the Infostop model with desired parameters
# %%
model = Infostop(
    r1=1,  # Max distance to consider points as stationary
    r2=1,  # Max distance to consider stationary points as connected
    min_staying_time=300,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=86400,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=True  # Set to True for more detailed output during processing
)

# %%
# Use fit_predict to process the data and get stop location labels
#  labels = model.fit_predict(example_data)
labels = model.fit_predict(data)
labels = model.fit_predict(df)
len(labels)

labs = pd.DataFrame(labels,columns=["label"])
labs.value_counts()

data["label"] = labels
#  data.loc[data["label"] != -1].to_csv("check_labels.csv",index=False)

#  data.to_csv("check_labels.csv",index=False)



# %%
