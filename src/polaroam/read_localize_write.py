import polars as pl
from polario.hive_dataset import HiveDataset
import os
import hashlib

data_dir = os.path.join("/","data","Berkeley","MX","data_quadrant")

df = pl.scan_parquet(os.path.join(data_dir,"filter_partioned","**","*.parquet"))
#  df.collect_schema()

columns_to_read = ["_c0", "_c2", "_c3", "_c5", "_c4"]
df = df.select(columns_to_read)
df = df.rename({"_c0":"uid", "_c2":"latitude", "_c3":"longitude", "_c5":"timestamp", "_c4":"error"})

#  df.collect_schema()

df = df.filter(pl.col("error") < 20)

df = df.with_columns([
    pl.from_epoch("timestamp", time_unit="s").dt.replace_time_zone("UTC").alias("utc_date")
]).with_columns([
    pl.col("utc_date").dt.convert_time_zone("America/Mexico_City").alias("date")
]).with_columns([
    pl.col("date").dt.replace_time_zone("UTC").dt.epoch(time_unit="s").alias("timestamp")
]).with_columns([
    pl.col("date").dt.date().alias("date_trunc").cast(pl.String)
])

ds = HiveDataset(os.path.join(data_dir,"filtered_localized"), partition_columns=["date_trunc"])

pl.Config.set_streaming_chunk_size(10000)

ds.write(df.collect(streaming=True))


###### Whole sample own subsample and partition

def create_big_int(incoming_data: str) -> int:
    """
    Create a unique big int from a string
    """
    data = str(incoming_data)
    sha256 = hashlib.sha256()
    sha256.update(data.encode())
    hash_bytes = sha256.digest()
    return int.from_bytes(hash_bytes, "big") % (2**63)

data_dir = os.path.join("/","data","Berkeley","MX","data_quadrant")
data_path = os.path.join(data_dir,"sample")
df = pl.scan_csv(os.path.join(data_path,"**","*.gz"), separator=",", has_header=False)

columns_to_read = ["column_1", "column_3", "column_4", "column_5", "column_6"]
df = df.select(columns_to_read)
df = df.rename({"column_1":"uid", "column_3":"latitude", "column_4":"longitude", "column_6":"timestamp", "column_5":"error"})

df = df.filter(pl.col("error") < 30)
df = df.with_columns(
    uid = create_big_int("uid"),
    timestamp = pl.col("timestamp") / 1000
).with_columns([
    pl.from_epoch("timestamp", time_unit="s").dt.replace_time_zone("UTC").alias("utc_date")
]).with_columns([
    pl.col("utc_date").dt.convert_time_zone("America/Mexico_City").alias("date")
]).with_columns([
    pl.col("date").dt.replace_time_zone("UTC").dt.epoch(time_unit="s").alias("timestamp")
]).with_columns([
    pl.col("date").dt.date().alias("date_trunc").cast(pl.String)
])

# Step 1: Group by `uid` and count distinct `date_trunc` values
uid_date_counts = (
    df.group_by("uid").agg(pl.col("date_trunc").n_unique().alias("date_trunc_count"))
)

# Step 2: Filter groups where the count is greater than 6
uids_with_more_than_6_dates = uid_date_counts.filter(pl.col("date_trunc_count") > 6)

# Step 3: Join the filtered result back with the original DataFrame
filtered_df = (
    df.join(uids_with_more_than_6_dates, on="uid", how="inner")
)

filtered_df = filtered_df.drop("date_trunc_count")

ds = HiveDataset(os.path.join(data_dir,"new_filtered_localized"), partition_columns=["date_trunc"])

pl.Config.set_streaming_chunk_size(10000)

ds.write(filtered_df.collect(streaming=True))
