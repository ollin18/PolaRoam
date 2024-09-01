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


#### Checking###

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
df = df.lazy()

model = models.Infostop(
    r1=10,  # Max distance to consider points as stationary
    r2=10,  # Max distance to consider stationary points as connected
    min_staying_time=900,  # Minimum time to consider a location as a stop (same unit as time in data)
    max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
    min_size=2,  # Minimum number of points to consider a stop
    distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
    verbose=False,  # Set to True for more detailed output during processing
    num_threads = 30,
    min_spacial_resolution=0
)

labels = model.fit_predict(df)
medians = model.compute_label_medians()
stop_locations = model.compute_dbscan()
df = stop_locations#.collect()
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################

# %%
#  df = pl.scan_parquet("/data_1/ollin/quadrant_stop_clusters/all_days_clusters.parquet")
df = pl.scan_parquet("~/data_quadrant/ollin_clusters/all_days_clusters.parquet")

# %%
df.collect_schema()

# %%
df = (
        df
        .with_columns(
            pl.from_epoch("start_timestamp", time_unit="s").alias("t_start")
            , pl.from_epoch("end_timestamp", time_unit="s").alias("t_end")
        )
        .with_columns(
            pl.col("t_start").dt.year().alias("year")
            , pl.col("t_start").dt.month().alias("month")
            , pl.col("t_start").dt.day().alias("day")
            , pl.col("t_start").dt.hour().alias("hour")
            , pl.col("t_start").dt.date().alias("date")
            , pl.col("t_start").dt.weekday().alias("weekday")
        )
    )


#  pl.Config.set_streaming_chunk_size(5)
#  df.select("t_start", "t_end", "start_timestamp","day","weekday").collect().head(5)

df = df.with_columns(duration = (pl.col("end_timestamp") - pl.col("start_timestamp")))

# %%
# Save the processed stop locations
#  output_file = "/data_1/ollin/quadrant_stop_clusters/all_days_clusters_with_dates_hours.parquet"
#  pl.Config.set_streaming_chunk_size(10000)
#  df.collect(streaming=True).write_parquet(output_file, use_pyarrow=True)
#
#
#  # %%
#  df = pl.scan_parquet(output_file)
#  df.collect_schema()
# %%
df = (
    df
    .with_columns(
        pl.lit('O').alias('location_type')
        , pl.lit(-1).alias('home_label')
        , pl.lit(-1).alias('work_label')
    )
)

# %%
df.collect_schema()

df = df.filter(pl.col("uid") == user)
# %%
start_hour_day = 6
end_hour_day = 22
home_period_window = 21
min_periods_over_window = 0.4
min_pings_home_cluster_label = 20

# Get potential homes initial

#  df = df.lazy()
home_tmp = (
        df.filter(
            (pl.col("hour") >= end_hour_day) | (pl.col("hour") <= start_hour_day)
        )
)

visits_durations = home_tmp.select('uid', 'stop_locations', 'date', 'duration', 'cluster_counts').group_by(['uid','stop_locations', 'date']).sum().sort('date')
#  home_tmp.select('uid', 'stop_locations', 'date', 'duration', 'cluster_counts').group_by(['uid','stop_locations']).sum().sort('date')

# This code gives the rolling on date

def rolling_dates(df, min_periods_over_window, home_period_window):
    min_periods = int(min_periods_over_window * home_period_window)

    df = df.sort("date").with_columns([
        pl.col("uid"),
        pl.col("duration").rolling_sum_by("date",f"{home_period_window}d", min_periods=min_periods).alias("duration_cum"),
        pl.col("cluster_counts").rolling_sum_by("date",f"{home_period_window}d", min_periods=min_periods).alias("counts_cum"),
    ])

    return df.select("uid", "date", "stop_locations", "duration_cum", "counts_cum").drop_nulls().unique()


output_schema = {
    "uid": pl.String,
    "date": pl.Date,
    "stop_locations": pl.Int64,
    "duration_cum": pl.Int64,
    "counts_cum": pl.Int64,
}

#  home_tmp.collect_schema()
#  first_uid_expr = home_tmp.select(pl.col("uid").first()).collect().get_column("uid")[0]
#
#  # Filter the LazyFrame using the extracted first uid
#  u1 = home_tmp.filter(pl.col("uid") == first_uid_expr)
#  u1.collect_schema()
#  rolling_dates(u1, min_periods_over_window, home_period_window).collect().head(5)
#
#  u1.group_by("stop_locations").map_groups(lambda df:
#                                           rolling_dates(df,
#                                                         min_periods_over_window,
#                                                         home_period_window)
#                                       , schema = output_schema
#                                      ).collect().sort("date").head(15)

df = df.sort("date")
#  tmp = df.select("date", "duration", "cluster_counts").rolling(index_column="date", period=f'{home_period_window}D').agg(pl.sum(pl.lit(min_periods_over_window)

#  df = df.lazy()
cumulative = (home_tmp.group_by(["uid", "stop_locations"]).map_groups(lambda df:
                                     rolling_dates(df,
                                                   min_periods_over_window,
                                                   home_period_window)
                                     , schema = output_schema
                                )
              )
visits_durations.collect_schema()
cumulative.collect_schema()

merged = visits_durations.join(
    cumulative,
    on=["uid", "date", "stop_locations"],
    how="left"
)

merged = merged.filter(pl.col("counts_cum") > min_pings_home_cluster_label)

merged.collect_schema()

#### Add home labels

####### For dynamic labeling ############
#  labels_tmp = merged.sort(by=["date", "duration_cum"], descending=True).unique(subset=["uid", "date","stop_locations"], keep="first").sort(by="duration_cum").select(["uid", "date","stop_locations"])
#  the_labels = labels_tmp.select(["uid", "stop_locations"]).unique().with_row_index("home_label")
#  labels_tmp = labels_tmp.join(the_labels, on=["uid","stop_locations"],how="left")
#  labels_tmp = labels_tmp.unique(subset=["uid", "date"],keep="first")
####### For static labeling, as it is one month we should keep this ############
labels_tmp = merged.sort(by=["uid", "date", "duration_cum"], descending=True).unique(subset=["uid", "stop_locations"], keep="first").sort(by="duration_cum", descending=True).select(["uid", "stop_locations"]).unique().with_row_index("home_label")
#  the_labels = labels_tmp.unique().with_row_index("home_label")
#  labels_tmp = labels_tmp.join(the_labels, on=["uid","stop_locations"],how="left")
#  labels_tmp = labels_tmp.unique(subset=["uid", "date"],keep="first")

joined = df.join(
    labels_tmp,
    on=["stop_locations"],
    how="left",  # Left join to keep all rows from A
    suffix="_new"  # Add suffix to differentiate 'label' columns from B
)
joined.collect_schema()
# Update 'label' in A with the correct values from B where 'label' in A is -1

updated = joined.with_columns(
    pl.when(pl.col("home_label_new").is_not_null())
    .then(pl.lit("H"))  # Use 'label_new' from B if it exists
    .otherwise(pl.col("location_type"))  # Keep original 'label' from A otherwise
    .alias("location_type"),
    pl.when(pl.col("home_label_new").is_not_null())
    .then(pl.col("home_label_new"))  # Use 'label_new' from B if it exists
    .otherwise(pl.col("home_label"))  # Keep original 'label' from A otherwise
    .alias("home_label")
).select(df.collect_schema().names())

updated.collect_schema()

#  user = (
#          org.loc[org["country"] == "BR"]
#          .groupby("uid")
#          .count()
#          .sort_values("datetime"
#                       ,ascending=False)
#          .reset_index()
#          .iloc[0,0])

# %%

#  u1 = updated.filter(pl.col("uid") == user)

u1 = updated.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("cluster_counts"))
    .alias("count")
)
us1 = u1.collect()

us1 = us1.group_by("stop_locations").map_groups(lambda df: df.with_row_index("height"))
us1 = us1.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("height"))
    .alias("height")
)

import pathlib
#  path: pathlib.Path = "~/Downloads/median_stops_u1_labels.csv"
path: pathlib.Path = "~/Downloads/homes_u1.csv"
us1.write_csv(path, separator=",")

