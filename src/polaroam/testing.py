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
#  df = pl.from_pandas(filtered_df)
df = pl.from_pandas(org)
df = df.lazy()
df = df.with_columns([
    (pl.col("datetime").dt.timestamp() / 1000000).round(0).cast(pl.Int64).alias("timestamp")
])
df = df.sort("timestamp")
# df = df.lazy()

# %%
model = models.Stopdetect(
    r1=5,  # Max distance to consider points as stationary
    r2=10,  # Max distance to consider stationary points as connected
    min_staying_time=900,  # Minimum time to consider a location as a stop (same unit as time in data)
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

stop_locations.schema
#  org = pd.read_parquet("../../data/veraset_movement_416.snappy.parquet")
mx = org.loc[org.country == "MX"]

# %%
# Instantiate the new HWEstimate model with the same parameters
hw_model = models.HWEstimate(
    r1=model._r1,
    r2=model._r2,
    min_staying_time=model._min_staying_time,
    max_time_between=model._max_time_between,
    min_size=model._min_size,
    distance_metric=model._distance_metric,
    verbose=model._verbose,
    num_threads=model._num_threads,
    min_spacial_resolution=model._min_spacial_resolution,
    tz="America/Mexico_City"  # Or another timezone if different from default
)

# Now you can use hw_model to perform the same tasks
labels = hw_model.fit_predict(df)
medians = hw_model.compute_label_medians()
stop_locations = hw_model.compute_dbscan()

# stop_locations = hw_model.prepare_labeling()

# %%
# hw_model._calculate_date_counts(stop_locations, 730)
# %%
hw = hw_model.detect_home(
    start_hour_day=7,
    end_hour_day=21,
    min_periods_over_window=0.5,
    span_period=0.5,
    total_days=730
)

hw.collect()

# %%
df = hw_model._hw_df

total_days = 730 #self._calculate_total_days(df)# self._total_days

# Filter for potential home time periods (nighttime or weekends)
home_tmp = df.filter(
    (pl.col("hour") >= 21) |
    (pl.col("hour") <= 7) |
    (pl.col("weekday").is_between(6, 7, closed="both"))
)

# %%
start_hour_day=0,
end_hour_day=6,
min_periods_over_window=0.5,
span_period=0.5,
total_days=730

combined_counts = hw_model._calculate_date_counts(home_tmp, total_days)
# combined_counts.collect()


filtered_clusters = hw_model._filter_clusters(combined_counts, min_periods_over_window, span_period)
filtered_clusters.collect()

# home_label = hw_model._label_locations(filtered_clusters, "home_label", 0, "home_label")
# home_label.collect()





# %%
hw = hw_model.detect_work(
    start_hour_day=8,
    end_hour_day=18,
    min_periods_over_window=0.5,
    span_period=0.5,
    total_days=730
)


# %%
hw = hw.collect()

# %%







#  import geopandas as gpd
#  mapa = gpd.read_file("~/Dropbox/enlace_hacia_documentos/Berkeley/informalidad/data/inegi/mgccpv/shp/m/conjunto_de_datos/09m.shp")
#  from shapely.geometry import MultiPolygon
#  combined_geometry = mapa.unary_union
#  if isinstance(combined_geometry, MultiPolygon):
#      periphery = [polygon.exterior for polygon in combined_geometry.geoms]
#  else:
#      periphery = [combined_geometry.exterior]
#  periphery_gdf = gpd.GeoDataFrame(geometry=periphery)
#  periphery_gdf = gpd.read_file("~/Downloads/Entidades_Federativas.shp")
#  periphery_gdf = periphery_gdf.loc[periphery_gdf["CVE_ENT"] == "09"]
#  periphery_gdf = periphery_gdf.to_crs("EPSG:4326")
#
#  points_gdf = gpd.GeoDataFrame(
#      mx,
#      geometry=gpd.points_from_xy(mx.longitude, mx.latitude),
#      crs=periphery_gdf.crs  # Ensure CRS is the same
#  )
#  points_within_periphery = gpd.sjoin(points_gdf, periphery_gdf, how="inner", predicate='within')

#  filtered_df = pd.DataFrame(points_within_periphery.drop(columns='geometry'))

user = (
        filtered_df
        .groupby("uid")
        .count()
        .sort_values("datetime"
                     ,ascending=False)
        .reset_index()
        .iloc[15,0])

# %%
u1 = stop_locations.filter(pl.col("uid") == stop_locations["uid"][0])

u1 = stop_locations.filter(pl.col("uid") == user)

u1 = u1.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("cluster_counts"))
    .alias("count")
)

u1 = u1.group_by("stop_locations").map_groups(lambda df: df.with_row_index("height"))

import pathlib
#  path: pathlib.Path = "~/Downloads/median_stops_u1_labels.csv"
path: pathlib.Path = "~/Downloads/median_stops_mx_labels.csv"
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
model = models.Polaroam(
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
model2 = models.Polaroam(
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


###################################################################
###################################################################
###################################################################

def assign_height(df):
    if df['stop_locations'].iloc[0] == -1:
        df['height'] = 1
    else:
        df['height'] = range(1, len(df) + 1)
    return df


cluster_labels = pd.read_parquet("~/data_quadrant/ollin_clusters/all_days_clusters.parquet")
list(cluster_labels)

users_l = (
        cluster_labels
        .groupby("uid")
        .count()
        .sort_values("start_timestamp"
                     ,ascending=False)
        .reset_index())

#  len(users_l.uid.unique())
user = users_l.iloc[400,0]


# %%
u1 = cluster_labels.loc[cluster_labels["uid"] == user]
u1['count'] = u1.apply(lambda row: 1 if row['stop_locations'] == -1 else row['cluster_counts'], axis=1)

u1 = u1.groupby('stop_locations').apply(assign_height).reset_index(drop=True)

path: pathlib.Path = "~/Downloads/quadrant_cluster_labels_1000.csv"
u1.to_csv(path, index=False)
