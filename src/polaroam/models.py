import numpy as np
import polars as pl
import utils
import importlib
importlib.reload(utils)
# from tqdm import tqdm

class NoStopsFoundException(Exception):
    pass

class BaseStopModel:
    """Base class for stop location models. Handles common functionality."""

    def __init__(self, distance_metric="haversine", verbose=False):
        self._distance_metric = distance_metric
        self._verbose = verbose
        self._is_fitted = False

    def _log(self, message):
        if self._verbose:
            print(message)


    def _data_assertions(self, data):
        def check_user_data(data, error_insert=""):
            # Ensure that 'latitude' and 'longitude' columns are present
            required_columns = ['latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in data.columns]
            assert not missing_columns, f"{error_insert}Missing columns: {', '.join(missing_columns)}"

            if 'timestamp' in data.columns:
                # Ensure timestamps are ordered
                assert np.array(data['timestamp'] == sorted(data['timestamp'])).all(), f"{error_insert}Timestamps must be ordered"

            if self._distance_metric == "haversine":
                # Ensure latitude and longitude are within valid ranges
                assert data['latitude'].min() > -90 and data['latitude'].max() < 90, f"{error_insert}Latitude must be between -90 and 90"
                assert data['longitude'].min() > -180 and data['longitude'].max() < 180, f"{error_insert}Longitude must be between -180 and 180"

        if self.multiuser:
            unique_uids = data.select('uid').unique().to_series().to_list()
            for u in unique_uids:
                error_insert = f"User {u}: "
                coords_u = data.filter(pl.col('uid') == u)
                check_user_data(coords_u, error_insert)
        else:
            check_user_data(data)

    def _fit_network(self, coords, r, weighted, weight_exponent, label_singleton, num_threads):
        ball_tree_result, counts = utils.query_neighbors(coords, r, self._distance_metric, weighted, num_threads)
        if weighted:
            node_idx_neighbors, node_idx_distances = ball_tree_result
        else:
            node_idx_neighbors, node_idx_distances = ball_tree_result, None

        labels = utils.label_network(node_idx_neighbors, node_idx_distances, counts, weight_exponent, label_singleton, self._distance_metric, self._verbose)

        return labels

    def _downsample(self, df, min_spacial_resolution):
        if min_spacial_resolution > 0:
            coords_df = (
                df.with_columns(
                    ((pl.col("latitude") / min_spacial_resolution).round() * min_spacial_resolution),
                    ((pl.col("longitude") / min_spacial_resolution).round() * min_spacial_resolution)
                ).with_columns(
                    pl.concat_list(["latitude", "longitude"]).alias("coords")
                )
            )
        else:
            coords_df = df.with_columns([
            pl.concat_list(["latitude", "longitude"]).alias('coords')
        ])

        coords_df = coords_df.with_row_index("index")

        unique_coords_df = (coords_df.group_by(["uid", "latitude", "longitude"], maintain_order=True)
                            .agg(pl.col('index')
                                , pl.col('stop_events') # this changed
                                , pl.len().alias("count")
                                , pl.col('coords').first()
                                , pl.col('start_timestamp')
                                , pl.col('end_timestamp')
                                )
                            )

        return unique_coords_df

    # def _calculate_total_days(self, df):
    #     # Aggregate to get min and max of the "t_start" column
    #     aggregated = df.select([
    #         pl.col("t_start").min().alias("min_date"),
    #         pl.col("t_start").max().alias("max_date")
    #     ])

    #     # Collect the result to get the min and max dates
    #     result = aggregated.collect()

    #     # Extract min_date and max_date from the result
    #     min_date = result["min_date"][0]
    #     max_date = result["max_date"][0]

    #     # Calculate the total days including both start and end dates
    #     total_days = (max_date - min_date).days + 1

    #     return total_days

    # def _calculate_date_counts(self, df, total_days):
    #     uid_date_counts = df.group_by("uid").agg(
    #         pl.col("date").n_unique().alias("total_dates"),
    #         pl.lit(total_days).alias("time_span")
    #     )

    #     cluster_date_counts = df.group_by(["uid", "stop_locations"]).agg(
    #         pl.col("date").n_unique().alias("cluster_dates")
    #     )

    #     combined_counts = cluster_date_counts.join(uid_date_counts, on="uid").with_columns([
    #         (pl.col("cluster_dates") / pl.col("total_dates")).alias("date_percentage"),
    #         (pl.col("cluster_dates") / pl.col("time_span")).alias("all_percentage")
    #     ])

    #     return combined_counts

    # def _filter_clusters(self, df, total_days, min_periods_over_window, span_period):
    #     total_days = _calculate_total_days(self, df)
    #     combined_counts = _calculate_date_counts(self, df, total_days)
    #     filtered_clusters =  combined_counts.filter(
    #         (pl.col("date_percentage") >= min_periods_over_window) &
    #         (pl.col("all_percentage") >= span_period)
    #     ).select(["uid", "stop_locations", "date_percentage", "all_percentage"])

    #     filtered_df = df.join(filtered_clusters, on=["uid", "stop_locations"], how="inner")
    #     return filtered_df

    # def _label_locations(self, df, label_column, label_value, new_label_column_name):
    #     label_df = df.sort(["date_percentage", "cluster_counts"], descending=True).unique(
    #         subset=["uid", "stop_locations"], keep="first"
    #     ).select("uid", "stop_locations").with_columns(
    #         pl.lit(label_value).alias(new_label_column_name)
    #     )

    #     return label_df


class Stopdetect(BaseStopModel):
    """Infostop model class, extending BaseStopModel."""
    def __init__(self, r1=10, r2=10, label_singleton=False, min_staying_time=300, max_time_between=86400, min_size=2, min_spacial_resolution=0, distance_metric="haversine", weighted=False, weight_exponent=1, verbose=False, coords_column='event_maps', num_threads=1, **kwargs):
        """
        Initialize HWEstimate with an optional timezone parameter and any other parameters for Stopdetect.

        Parameters:
        - kwargs: Other parameters to initialize the Stopdetect class.
        """
        super().__init__(**kwargs)
        self._r1 = r1
        self._r2 = r2
        self._label_singleton = label_singleton
        self._min_staying_time = min_staying_time
        self._max_time_between = max_time_between
        self._min_size = min_size
        self._min_spacial_resolution = min_spacial_resolution
        self._coords_column = coords_column
        self._weighted = weighted
        self._weight_exponent = weight_exponent
        self._num_threads = num_threads

    # def __init__(self, r1=10, r2=10, label_singleton=False, min_staying_time=300, max_time_between=86400, min_size=2, min_spacial_resolution=0, distance_metric="haversine", weighted=False, weight_exponent=1, verbose=False, coords_column='event_maps', num_threads=1):
    #     super().__init__(distance_metric, verbose)
    #     self._r1 = r1
    #     self._r2 = r2
    #     self._label_singleton = label_singleton
    #     self._min_staying_time = min_staying_time
    #     self._max_time_between = max_time_between
    #     self._min_size = min_size
    #     self._min_spacial_resolution = min_spacial_resolution
    #     self._coords_column = coords_column
    #     self._weighted = weighted
    #     self._weight_exponent = weight_exponent
    #     self._num_threads = num_threads

    def fit_predict(self, data):
        # progress = tqdm if self._verbose else utils.pass_func

        # Check if 'uid' is in the column names
        column_names = data.collect_schema().names()
        if 'uid' in column_names:
            # Select the unique count of 'uid'
            #  unique_uid_count = data.select(pl.col("uid").n_unique().alias("unique_uid_count")).collect()
            # Extract the value and perform the comparison
            #  self.multiuser = unique_uid_count["unique_uid_count"][0] > 1
            self.multiuser = True
        else:
            self.multiuser = False

        if self.multiuser:
            # Group the data by 'uid' for multiuser set
            grouped_data = data.group_by('uid', maintain_order=True)
        else:
            # Treat as single user data
            grouped_data = data.with_columns(pl.lit("single_user").alias("uid")).group_by("uid", maintain_order=True)
        # self._data_assertions(data)
        # Define the schema for the output DataFrame
        output_schema = {
            "uid": pl.String,
            "stop_events": pl.Int64,
            "event_maps": pl.Array(pl.Float64, shape=(2,)),
            "timestamp": pl.Int64,
        }
        # Use apply to handle the stationary events per group in parallel
        results = grouped_data.map_groups(
            lambda df: pl.DataFrame(
                utils.get_stationary_events(
                    df.select(['uid', 'latitude', 'longitude', 'timestamp']),
                    self._r1,
                    self._min_size,
                    self._min_staying_time,
                    self._max_time_between,
                    self._distance_metric
                )
            )
            , schema = output_schema
        )

        # if ((results.filter(pl.col('stop_events') == -1).shape[0]) == results.shape[0]):
        # if ((results.filter(pl.col('stop_events') == -1).select(pl.len()).collect().item()) == results.select(pl.len()).collect().item()):
        #     raise NoStopsFoundException("No stop events found. Check parameters.")

        self._results = results
        self._is_fitted = True

        return self._results

    def compute_label_medians(self):
        self._fitted_assertion()
        labels_and_coords = self._results.filter(pl.col("stop_events") != -1)
        labels_and_coords = labels_and_coords.with_columns(pl.col("event_maps").arr.get(0).alias("latitude"),
                        pl.col("event_maps").arr.get(1).alias("longitude"))

        self._median_coords = (labels_and_coords
                               .group_by(["uid", "stop_events"], maintain_order=True)
                               .agg(pl.col("latitude").median()
                                    , pl.col("longitude").median()
                                    , pl.col("timestamp").min().alias("start_timestamp")
                                    , pl.col("timestamp").max().alias("end_timestamp")
                                )
                            )
        return self._median_coords

    def compute_infomap(self):
        self._stat_coords = self._downsample(self._median_coords, self._min_spacial_resolution)

        # self._log(f"Downsampling to {len(self._stat_coords)} unique coordinates.")
        # self._stat_labels = self._fit_network(self._stat_coords, self._r2, self._weighted, self._weight_exponent, self._label_singleton, self._num_threads)

        self._stat_labels = (self._stat_coords
                                .group_by("uid", maintain_order=True)
                                .map_groups(lambda df: pl.DataFrame(
                                    self._fit_network(df
                                                        , self._r2
                                                        , self._weighted
                                                        , self._weight_exponent
                                                        , self._label_singleton
                                                        , self._num_threads))))

        labels_df = (self._stat_coords.select(pl.col("coords").alias("stat_coords")
                                              , pl.col("index").alias("inverse_indices")
                                              )
                                      .with_columns(labels = self._stat_labels.to_numpy().flatten())
                     ).explode(pl.col("inverse_indices")).sort("inverse_indices")

        stop_labels = self._median_coords.with_columns((pl.Series(name="stop_labels", values=labels_df['labels'])) )
        self._stop_labels = stop_labels

        return self._stop_labels

    def compute_dbscan(self):
        self._stat_coords = self._downsample(self._median_coords, self._min_spacial_resolution)

        output_schema = {"uid":pl.String
                         ,"stop_events":pl.Int64
                         ,"inverse_indices":pl.List(pl.UInt32)
                         ,"latitude":pl.Float64
                         ,"longitude":pl.Float64
                         ,"start_timestamp":pl.Int64
                         ,"end_timestamp":pl.Int64
                         ,"stop_locations":pl.Int64
                         }
        self._stat_labels = (self._stat_coords
                                .group_by("uid", maintain_order=True)
                                .map_groups(lambda df: pl.DataFrame({
                                    "uid":df.select(pl.col('uid'))
                                    ,"stop_events":df.select(pl.col("stop_events"))
                                    ,"inverse_indices":df.select(pl.col("index"))
                                    ,"latitude":df.select(pl.col('latitude'))
                                    ,"longitude":df.select(pl.col('longitude'))
                                    ,"start_timestamp":df.select(pl.col("start_timestamp"))
                                    ,"end_timestamp":df.select(pl.col('end_timestamp'))
                                    ,"stop_locations":utils.cluster_dbscan(df
                                                        , self._r2
                                                        , self._distance_metric
                                                        , self._num_threads)
                                                    })
                                                , schema = output_schema
                                            )
                                        )

        stop_labels = self._stat_labels.explode(["inverse_indices","stop_events","start_timestamp","end_timestamp"]).sort("inverse_indices")

        output_schema = {"uid":pl.String
                         ,"stop_locations":pl.Int64
                         ,"cluster_counts":pl.UInt32
                         ,"cluster_latitude":pl.Float64
                         ,"cluster_longitude":pl.Float64
                         }
        stop_medoid = (stop_labels
                       .group_by(["uid","stop_locations"], maintain_order=True)
                       .map_groups(lambda df: pl.DataFrame({
                                    "uid":df.select(pl.col('uid').first())
                                    ,"stop_locations":df.select(pl.col("stop_locations").first())
                                    ,"cluster_counts":df.select(pl.len())
                                    ,"cluster_latitude":df.select(pl.col("latitude")).median()
                                    ,"cluster_longitude":df.select(pl.col("longitude")).median()
                                })
                            , schema = output_schema
                            )
                            .with_columns(
                                cluster_counts = pl.when(pl.col("stop_locations") == -1)
                                .then(1)
                                .otherwise(pl.col("cluster_counts"))
                            )
                       )
        # stop_medoid = (
        #                 stop_labels
        #                 .group_by(["uid", "stop_locations"], maintain_order=True)
        #                 .agg([
        #                     pl.col("uid").first().alias("uid"),
        #                     pl.col("stop_locations").first().alias("stop_locations"),
        #                     pl.when(pl.col("stop_locations").first() == -1)
        #                     .then(pl.lit(1))
        #                     .otherwise(pl.col("stop_locations").count())
        #                     .alias("cluster_counts"),
        #                     pl.col("latitude").median().alias("cluster_latitude"),
        #                     pl.col("longitude").median().alias("cluster_longitude")
        #                 ])
        #             )

        self._stop_labels = stop_labels.join(stop_medoid, on=["uid", "stop_locations"], how="left")

        return self._stop_labels

    def _fitted_assertion(self):
        assert self._is_fitted, "Model must be fitted before this operation."


class HWEstimate(Stopdetect):
    def __init__(self,
                 start_hour_day=7,
                 end_hour_day=21,
                 min_periods_over_window=0.5,
                 span_period=0.5,
                 total_days=30,
                 convert_tz=False,
                 tz="UTC", **kwargs):
        """
        Initialize HWEstimate with an optional timezone parameter and any other parameters for Stopdetect.

        Parameters:
        - conver_tz: Logic variable to convert the time-zone if original data is not
          localized.
        - tz: Timezone for datetime operations, default is "UTC".
        - kwargs: Other parameters to initialize the Stopdetect class.
        """
        super().__init__(**kwargs)
        self._start_hour_day=start_hour_day
        self._end_hour_day=end_hour_day
        self._min_periods_over_window=min_periods_over_window
        self._span_period=span_period
        self._total_days=total_days
        self._convert_tz = convert_tz  # Store the timezone parameter
        self._tz = tz  # Store the timezone parameter

    def prepare_labeling(self, df):
        """
        Prepare the dataframe for home and work location labeling.

        Parameters:
        - df: Polars DataFrame containing location data with start and end timestamps.

        Returns:
        - df: Updated Polars DataFrame with additional columns for labeling.
        """
        # Convert timestamps to datetime and set the timezone
        df = df.with_columns([
            pl.from_epoch("start_timestamp", time_unit="s").alias("t_start"),
            pl.from_epoch("end_timestamp", time_unit="s").alias("t_end"),
        ])
        if self._convert_tz:
            df = df.with_columns([
                pl.col("t_start").dt.convert_time_zone(self._tz).alias("t_start"),
                pl.col("t_end").dt.convert_time_zone(self._tz).alias("t_end"),
            ])

        # Extract date and time components
        df = df.with_columns([
            pl.col("t_start").dt.year().alias("year"),
            pl.col("t_start").dt.month().alias("month"),
            pl.col("t_start").dt.day().alias("day"),
            pl.col("t_start").dt.hour().alias("hour"),
            pl.col("t_start").dt.date().alias("date"),
            pl.col("t_start").dt.weekday().alias("weekday"),
        ])

        # Calculate duration and initialize labels
        df = df.with_columns([
            (pl.col("end_timestamp") - pl.col("start_timestamp")).alias("duration"),
            pl.lit('O').alias('location_type'),
            pl.lit(-1).alias('home_label'),
            pl.lit(-1).alias('work_label'),
        ])

        # self._total_days = self._calculate_total_days(df)
        self._hw_df = df
        return self._hw_df

    def detect_home(self):
        """
        Identify potential home locations based on recurring visits.

        Parameters:
        - df: Polars DataFrame containing location data.
        - start_hour_day: The hour indicating the start of the day (for detecting home stays).
        - end_hour_day: The hour indicating the end of the day (for detecting home stays).
        - min_periods_over_window: Minimum percentage of unique dates required to consider a location as home.
        - span_period: Minimum percentage of time span required to consider a location as home.
        - total_days: Total number of days to consider for calculating percentages. If None, use the span of the entire dataset.

        Returns:
        - df: Updated Polars DataFrame with home labels.
        """

        # Calculate total_days if not provided
        # if self._total_days is None:
        #     total_days = utils.calculate_total_days(self._hw_df)# self._total_days
        # else:
        #     total_days = self._total_days

        # Filter for potential home time periods (nighttime or weekends)
        home_tmp = self._hw_df.filter(
            (pl.col("hour") >= self._end_hour_day) |
            (pl.col("hour") <= self._start_hour_day) |
            (pl.col("weekday").is_between(6, 7, closed="both")) &
            (pl.col("stop_locations") != -1)
        )

        # combined_counts = self._calculate_date_counts(home_tmp, total_days)

        filtered_clusters = utils.filter_clusters(home_tmp, self._total_days, self._min_periods_over_window, self._span_period)

        home_label = utils.label_locations(filtered_clusters, "home_label", 0, "home_label")

        updated = self._hw_df.join(home_label, on=["uid", "stop_locations"], how="left", suffix="_new").with_columns([
            pl.when(pl.col("home_label_new").is_not_null())
            .then(pl.lit("H"))
            .otherwise(pl.col("location_type"))
            .alias("location_type"),
            pl.when(pl.col("home_label_new").is_not_null())
            .then(pl.col("home_label_new"))
            .otherwise(pl.col("home_label"))
            .alias("home_label")
        ]).select(self._hw_df.collect_schema().names())

        self._hw_df = updated
        self._home_detected = True  # Set the flag to indicate that home detection is complete
        return self._hw_df

    def detect_work(self):
        """
        Identify potential work locations based on recurring visits.

        Parameters:
        - start_hour_day: The hour indicating the start of the workday (default is 8 AM).
        - end_hour_day: The hour indicating the end of the workday (default is 6 PM).
        - min_periods_over_window: Minimum percentage of unique dates required to consider a location as work.
        - span_period: Minimum percentage of time span required to consider a location as work.
        - total_days: Total number of days to consider for calculating percentages. If None, use the span of the entire dataset.

        Returns:
        - Updated Polars DataFrame with work labels.
        """
        # Assert that home detection has been done
        assert self._home_detected, "Home detection must be performed before work detection."

        # # Calculate total_days if not provided
        # if self._total_days is None:
        #     total_days = utils.calculate_total_days(self._hw_df)# self._total_days
        # else:
        #     total_days = self._total_days

        # Filter for potential work time periods (work hours on weekdays, excluding home locations)
        work_tmp = self._hw_df.filter(
            ((pl.col("hour") >= self._start_hour_day) & (pl.col("hour") <= self._end_hour_day)) &
            (pl.col("weekday").is_between(1, 5, closed="both")) &
            (pl.col("location_type") != "H") &
            (pl.col("stop_locations") != -1)
        )

        filtered_clusters = utils.filter_clusters(work_tmp, self._total_days, self._min_periods_over_window, self._span_period)

        work_label = utils.label_locations(filtered_clusters, "work_label", 0, "work_label")

        updated = self._hw_df.join(work_label, on=["uid", "stop_locations"], how="left", suffix="_new").with_columns([
            pl.when(pl.col("work_label_new").is_not_null())
            .then(pl.lit("W"))
            .otherwise(pl.col("location_type"))
            .alias("location_type"),
            pl.when(pl.col("work_label_new").is_not_null())
            .then(pl.col("work_label_new"))
            .otherwise(pl.col("work_label"))
            .alias("work_label")
        ]).select(self._hw_df.collect_schema().names())

        self._hw_df = updated
        return self._hw_df
