# %%
import os
#  os.environ['POLARS_MAX_THREADS'] = '20'
import pandas as pd
import polars as pl
import polaroam
import models
from models import Infostop
import numpy as np
import utils
import time

def rolling_dates(df, min_periods_over_window, home_period_window):
    min_periods = int(min_periods_over_window * home_period_window)

    df = df.sort("date").with_columns([
        pl.col("uid"),
        pl.col("duration").rolling_sum_by("date",f"{home_period_window}d", min_periods=min_periods).alias("duration_cum"),
        pl.col("cluster_counts").rolling_sum_by("date",f"{home_period_window}d", min_periods=min_periods).alias("counts_cum"),
    ])

    return df.select("uid", "date", "stop_locations", "duration_cum", "counts_cum").drop_nulls().unique()



#### Checking###

#  org = pd.read_parquet("../../data/veraset_movement_416.snappy.parquet")
#  # df2 = pl.read_parquet("../../data/veraset_movement_416.snappy.parquet")
#
#  # %%
#  #  df = pl.from_pandas(filtered_df)
#  df = pl.from_pandas(org)
#  #  df = df.lazy()
#  df = df.with_columns([
#      (pl.col("datetime").dt.timestamp() / 1000000).round(0).cast(pl.Int64).alias("timestamp")
#  ])
#  df = df.sort("timestamp")
#  df = df.lazy()
#
#  model = models.Infostop(
#      r1=10,  # Max distance to consider points as stationary
#      r2=10,  # Max distance to consider stationary points as connected
#      min_staying_time=900,  # Minimum time to consider a location as a stop (same unit as time in data)
#      max_time_between=3600,  # Max time between points to consider them in the same stop (same unit as time in data)
#      min_size=2,  # Minimum number of points to consider a stop
#      distance_metric="haversine",  # Distance metric, 'haversine' for geo-coordinates
#      verbose=False,  # Set to True for more detailed output during processing
#      num_threads = 30,
#      min_spacial_resolution=0
#  )
#
#  labels = model.fit_predict(df)
#  medians = model.compute_label_medians()
#  stop_locations = model.compute_dbscan()
#  df = stop_locations#.collect()
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
#  df = pl.scan_parquet("~/data_quadrant/ollin_clusters/all_days_clusters.parquet")

# %%
#  df.collect_schema()

# %%
df = pl.scan_parquet("~/data_quadrant/ollin_clusters/all_days_clusters.parquet")
def prepare_labeling(df):
    df = (
            df
            .with_columns(
                pl.from_epoch("start_timestamp", time_unit="s").alias("t_start")
                , pl.from_epoch("end_timestamp", time_unit="s").alias("t_end")
            )
            .with_columns(
                pl.col("t_start").dt.convert_time_zone("America/Mexico_City").alias("t_start"),
                pl.col("t_end").dt.convert_time_zone("America/Mexico_City").alias("t_end")
            )
            .with_columns(
                pl.col("t_start").dt.replace_time_zone("UTC").alias("t_start"),
                pl.col("t_end").dt.replace_time_zone("UTC").alias("t_end")
            )
            .with_columns(
                pl.col("t_start").dt.convert_time_zone("America/Mexico_City").alias("t_start"),
                pl.col("t_end").dt.convert_time_zone("America/Mexico_City").alias("t_end")
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

    df = df.with_columns(duration = (pl.col("end_timestamp") - pl.col("start_timestamp")))

    df = (
        df
        .with_columns(
            pl.lit('O').alias('location_type')
            , pl.lit(-1).alias('home_label')
            , pl.lit(-1).alias('work_label')
        )
    )

    return df

def cheap_home(df):
    home_tmp = (
            df.filter(
                (
                    (pl.col("hour") >= end_hour_day) | (pl.col("hour") <= start_hour_day)
                ) | (pl.col("weekday").is_between(6, 7, closed="both"))
            )
    )

    min_periods_over_window = 0.5
    span_period = 0.5


    uid_date_counts = home_tmp.group_by("uid").agg(pl.col("date").n_unique().alias("total_dates"), pl.lit(31).alias("time_span"))

    cluster_date_counts = home_tmp.group_by(["uid", "stop_locations"]).agg(pl.col("date").n_unique().alias("cluster_dates"))

    combined_counts = cluster_date_counts.join(uid_date_counts, on="uid")

    combined_counts = combined_counts.with_columns(
        (pl.col("cluster_dates") / pl.col("total_dates")).alias("date_percentage")
        , (pl.col("cluster_dates") / pl.col("time_span")).alias("all_percentage")
    )

    filtered_clusters = combined_counts.filter(
        (pl.col("date_percentage") >= min_periods_over_window) & (pl.col("all_percentage") >= span_period)
    ).select(["uid", "stop_locations", "date_percentage", "all_percentage"])

    filtered_df = df.join(filtered_clusters, on=["uid", "stop_locations"], how="inner")

    home_label = filtered_df.sort(["date_percentage","cluster_counts"], descending=True).unique(subset=["uid", "stop_locations"], keep="first").select("uid","stop_locations")
    home_label = home_label.with_columns(home_label = pl.lit(0))

    joined = df.join(
        home_label,
        on=["uid", "stop_locations"],
        how="left",
        suffix="_new"
    )
    joined.collect_schema()

    updated = joined.with_columns(
        pl.when(pl.col("home_label_new").is_not_null())
        .then(pl.lit("H"))
        .otherwise(pl.col("location_type"))
        .alias("location_type"),
        pl.when(pl.col("home_label_new").is_not_null())
        .then(pl.col("home_label_new"))
        .otherwise(pl.col("home_label"))
        .alias("home_label")
    ).select(df.collect_schema().names())

    return updated

def cheap_work(df):

    work_tmp = df.filter(
    (
        (pl.col("hour") >= start_hour_day) | (pl.col("hour") <= end_hour_day)
    ) & (pl.col("weekday").is_between(1, 5, closed="both"))
    & (pl.col("location_type") != "H")
    )

    min_periods_over_window = 0.5
    span_period = 0.5


    uid_date_counts = work_tmp.group_by("uid").agg(pl.col("date").n_unique().alias("total_dates"), pl.lit(31).alias("time_span"))

    cluster_date_counts = work_tmp.group_by(["uid", "stop_locations"]).agg(pl.col("date").n_unique().alias("cluster_dates"))

    combined_counts = cluster_date_counts.join(uid_date_counts, on="uid")

    combined_counts = combined_counts.with_columns(
        (pl.col("cluster_dates") / pl.col("total_dates")).alias("date_percentage")
        , (pl.col("cluster_dates") / pl.col("time_span")).alias("all_percentage")
    )

    filtered_clusters = combined_counts.filter(
        (pl.col("date_percentage") >= min_periods_over_window) & (pl.col("all_percentage") >= span_period)
    ).select(["uid", "stop_locations", "date_percentage","all_percentage"])

    #  filtered_duration = df.group_by(["uid", "stop_locations"]).agg(pl.col("duration").sum().alias("total_duration")).select("uid","stop_locations","total_duration")

    filtered_df = df.join(filtered_clusters, on=["uid", "stop_locations"], how="inner")#.join(filtered_duration, on=["uid", "stop_locations"])

    work_label = filtered_df.sort(["date_percentage","cluster_counts"], descending=True).unique(subset=["uid", "stop_locations"], keep="first").select("uid","stop_locations")
    work_label = work_label.with_columns(work_label = pl.lit(0))

    joined = df.join(
        work_label,
        on=["uid", "stop_locations"],
        how="left",
        suffix="_new"
    )
    joined.collect_schema()

    updated = joined.with_columns(
        pl.when(pl.col("work_label_new").is_not_null())
        .then(pl.lit("W"))
        .otherwise(pl.col("location_type"))
        .alias("location_type"),
        pl.when(pl.col("work_label_new").is_not_null())
        .then(pl.col("work_label_new"))
        .otherwise(pl.col("work_label"))
        .alias("work_label")
    ).select(df.collect_schema().names())

    return updated



def get_home_locations(df):
    home_tmp = (
            df.filter(
                (pl.col("hour") >= end_hour_day) | (pl.col("hour") <= start_hour_day)
            )
    )

    visits_durations = home_tmp.select('stop_locations', 'date', 'duration', 'cluster_counts').group_by(['stop_locations', 'date']).sum().sort('date')

    #  output_schema = {
    #      "date": pl.Date,
    #      "stop_locations": pl.Int64,
    #      "duration_cum": pl.Int64,
    #      "counts_cum": pl.Int64,
    #  }
    cumulative = (home_tmp.group_by("stop_locations").map_groups(lambda df:
                                         rolling_dates(df,
                                                       min_periods_over_window,
                                                       home_period_window)
                                         #  , schema = output_schema
                                    )
                  )

    merged = visits_durations.join(
        cumulative,
        on=["date", "stop_locations"],
        how="left"
    )

    merged = merged.filter(pl.col("counts_cum") > min_pings_home_cluster_label)

    return merged

start_hour_day = 8
end_hour_day = 21
home_period_window = 21
min_periods_over_window = 0.6
min_pings_home_cluster_label = 20

df.collect_schema()

df = prepare_labeling(df)
import pathlib
path: pathlib.Path = "~/data_quadrant/home_work/all_users_prepared_data.parquet"
start = time.time()
pl.Config.set_streaming_chunk_size(10000)
df.collect(streaming=True).write_parquet(path, use_pyarrow=True)
end = time.time()
print(end - start)



df = pl.scan_parquet("~/data_quadrant/home_work/all_users_prepared_data.parquet")
df = cheap_home(df)
df = cheap_work(df)

u1 = df.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("cluster_counts"))
    .alias("count")
)

#  us1= us1.drop("height")
out_schema = pl.Schema([('height', pl.Int64),
        ('uid', pl.String),
        ('stop_events', pl.Int64),
        ('inverse_indices', pl.Int64),
        ('latitude', pl.Float64),
        ('longitude', pl.Float64),
        ('start_timestamp', pl.Int64),
        ('end_timestamp', pl.Int64),
        ('stop_locations', pl.Int64),
        ('cluster_counts', pl.Int64),
        ('cluster_latitude', pl.Float64),
        ('cluster_longitude', pl.Float64),
        ('t_start', pl.String),
        ('t_end', pl.String),
        ('year', pl.Int64),
        ('month', pl.Int64),
        ('day', pl.Int64),
        ('hour', pl.Int64),
        ('date', pl.String),
        ('weekday', pl.Int64),
        ('duration', pl.Int64),
        ('location_type', pl.String),
        ('home_label', pl.Int64),
        ('work_label', pl.Int64),
        ('count', pl.Int64)])
us1 = u1.group_by("uid","stop_locations").map_groups(lambda df: df.with_row_index("height"), schema=out_schema)
us1 = us1.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("height"))
    .alias("height")
)

import pathlib
#  path: pathlib.Path = "~/Downloads/median_stops_u1_labels.csv"
path: pathlib.Path = "~/data_quadrant/home_work/all_users_home_work_by_counts.parquet"

start = time.time()
pl.Config.set_streaming_chunk_size(10000)
us1.collect(streaming=True).write_parquet(path, use_pyarrow=True)
end = time.time()
print(end - start)

#  us1.write_csv(path, separator=",")

us1 = pl.read_parquet("~/data_quadrant/home_work/all_users_home_work_by_counts.parquet")


only_hw = us1.select("uid", "location_type", "cluster_latitude", "cluster_longitude")
only_hw = only_hw.filter(pl.col("location_type") != "O").unique(subset=["uid","location_type"])

path: pathlib.Path = "~/data_quadrant/home_work/home_work_locations_by_counts.csv"
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

path: pathlib.Path = "~/data_quadrant/home_work/hw_arcs_by_counts.csv"
wide_df.write_csv(path)




##############################################################################
##############################################################################
##############################################################################

pl.Schema([('height', pl.Int64),
        ('uid', pl.String),
        ('stop_events', pl.Int64),
        ('inverse_indices', pl.Int64),
        ('latitude', pl.Float64),
        ('longitude', pl.Float64),
        ('start_timestamp', pl.Int64),
        ('end_timestamp', pl.Int64),
        ('stop_locations', pl.Int64),
        ('cluster_counts', pl.Int64),
        ('cluster_latitude', pl.Float64),
        ('cluster_longitude', pl.Float64),
        ('t_start', pl.String),
        ('t_end', pl.String),
        ('year', pl.Int64),
        ('month', pl.Int64),
        ('day', pl.Int64),
        ('hour', pl.Int64),
        ('date', pl.String),
        ('weekday', pl.Int64),
        ('duration', pl.Int64),
        ('location_type', pl.String),
        ('home_label', pl.Int64),
        ('work_label', pl.Int64),
        ('count', pl.Int64)])

#### Cheap detection


#  pl.Config.set_streaming_chunk_size(5)
#  df.select("t_start", "t_end", "start_timestamp","day","weekday").collect().head(5)


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

# %%



home_tmp = (
        df.filter(
            (pl.col("hour") >= end_hour_day) | (pl.col("hour") <= start_hour_day)
        )
)

visits_durations = home_tmp.select('uid', 'stop_locations', 'date', 'duration', 'cluster_counts').group_by(['uid','stop_locations', 'date']).sum().sort('date')
#  home_tmp.select('uid', 'stop_locations', 'date', 'duration', 'cluster_counts').group_by(['uid','stop_locations']).sum().sort('date')

# This code gives the rolling on date


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

updated = df.collect()
u1 = updated.with_columns(
    pl.when(pl.col("stop_locations") == -1)
    .then(1)
    .otherwise(pl.col("cluster_counts"))
    .alias("count")
)
us1 = u1#.collect()

#  us1= us1.drop("height")
us1 = us1.group_by("uid","stop_locations").map_groups(lambda df: df.with_row_index("height"))
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

