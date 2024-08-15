import numpy as np
import polars as pl
import utils
import importlib
importlib.reload(utils)
from tqdm import tqdm

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
        coords = df.with_columns([
            pl.concat_list(["latitude", "longitude"]).alias('coords')
        ])
        if min_spacial_resolution > 0:
            # Extract the coords column as a Polars Series and convert it to a NumPy array
            coords_vals = coords['coords'].to_numpy()
            # coords_vals = np.around(coords_vals / min_spacial_resolution) * min_spacial_resolution
            coords_vals = np.array([np.around(coord / min_spacial_resolution) * min_spacial_resolution for coord in coords_vals])
            # Convert the NumPy array back to a Polars DataFrame or Series
            coords_df = coords.with_columns([pl.Series('coords', coords_vals)])
            coords_df = (coords_df.with_columns(pl.col('coords').arr.get(0).alias("latitude")
                                            , pl.col('coords').arr.get(1).alias("longitude")
                                            )
                                        )
        else:
            coords_df = coords
        
        coords_df = coords_df.with_row_index("index")
        
        unique_coords_df = (coords_df.group_by(["uid", "latitude", "longitude"], maintain_order=True)
                            .agg(pl.col('index')
                                 , pl.len().alias("count")
                                 , pl.col('coords').first()
                                 )
                                )
        return unique_coords_df

class Infostop(BaseStopModel):
    """Infostop model class, extending BaseStopModel."""

    def __init__(self, r1=10, r2=10, label_singleton=False, min_staying_time=300, max_time_between=86400, min_size=2, min_spacial_resolution=0, distance_metric="haversine", weighted=False, weight_exponent=1, verbose=False, coords_column='event_maps', num_threads=1):
        super().__init__(distance_metric, verbose)
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

    def fit_predict(self, data):
        progress = tqdm if self._verbose else utils.pass_func

        self.multiuser = 'uid' in data.columns and data['uid'].n_unique() > 1

        if self.multiuser:
            # Group the data by 'uid' for multiuser set
            grouped_data = data.group_by('uid', maintain_order=True)
        else:
            # Treat as single user data
            grouped_data = data.with_columns(pl.lit("single_user").alias("uid")).group_by("uid", maintain_order=True)
        self._data_assertions(data)

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
        )

        if ((results.filter(pl.col('stop_events') == -1).shape[0]) == results.shape[0]):
            raise NoStopsFoundException("No stop events found. Check parameters.")
        
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

        self._log(f"Downsampling to {len(self._stat_coords)} unique coordinates.")
        self._stat_labels = self._fit_network(self._stat_coords, self._r2, self._weighted, self._weight_exponent, self._label_singleton, self._num_threads)

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

    def _fitted_assertion(self):
        assert self._is_fitted, "Model must be fitted before this operation."


class SpatialInfomap(BaseStopModel):
    """Spatial Infomap class, extending BaseStopModel."""

    def __init__(self, r2=10, label_singleton=True, min_spacial_resolution=0, distance_metric="haversine", weighted=False, weight_exponent=1, verbose=False):
        super().__init__(distance_metric, verbose)
        self._r2 = r2
        self._label_singleton = label_singleton
        self._min_spacial_resolution = min_spacial_resolution
        self._weighted = weighted
        self._weight_exponent = weight_exponent

    def fit_predict(self, data):
        self._data = data
        self._data_assertions([data])

        self._stat_coords, inverse_indices, self._counts = self._downsample(self._data, self._min_spacial_resolution)
        self._log(f"Downsampling to {len(self._stat_coords)} unique coordinates.")
        self._stat_labels = self._fit_network(self._stat_coords, self._r2, self._weighted, self._weight_exponent, self._label_singleton)
        self._labels = self._stat_labels[inverse_indices]
        self._is_fitted = True
        return self._labels

