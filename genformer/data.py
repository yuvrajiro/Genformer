import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from darts import TimeSeries, concatenate
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from darts.dataprocessing.transformers import Scaler
from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import norm_freq_str

def gluonts_item_to_darts_mv(item, freq: str) -> TimeSeries:
    start = item["start"].to_timestamp() if hasattr(item["start"], "to_timestamp") else pd.Timestamp(item["start"])
    target = np.asarray(item["target"])
    if target.ndim != 2:
        raise ValueError(f"Expected multivariate target with ndim=2, got shape {target.shape}")

    values = target.T 
    times = pd.date_range(start=start, periods=values.shape[0], freq=freq)
    cols = [f"dim_{i}" for i in range(values.shape[1])]

    return TimeSeries.from_times_and_values(times, values, columns=cols)

def lag_covs_from_scaled_target(ts_sc: TimeSeries, lags=(1,24,168)) -> TimeSeries:
    shifted = []
    for L in lags:
        s = ts_sc.shift(L).with_columns_renamed(
            ts_sc.components, [f"{c}_lag{L}" for c in ts_sc.components]
        )
        shifted.append(s)

    common = shifted[0]
    for s in shifted[1:]:
        common = common.slice_intersect(s)
    shifted = [s.slice_intersect(common) for s in shifted]
    return concatenate(shifted, axis=1)

def fourier_from_index_min(idx) -> TimeSeries:
    minute = idx.minute.to_numpy()
    hour   = idx.hour.to_numpy()
    dow    = idx.dayofweek.to_numpy()

    X = np.vstack([
        np.sin(2 * np.pi * minute / 60.0),
        np.cos(2 * np.pi * minute / 60.0),
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
    ]).T

    return TimeSeries.from_times_and_values(
        idx,
        X,
        columns=["min_sin", "min_cos", "h_sin", "h_cos", "dow_sin", "dow_cos"]
    )

def fourier_from_index_day(idx) -> TimeSeries:
    dow  = idx.dayofweek.to_numpy()
    X = np.vstack([
        np.sin(2*np.pi*dow/7.0),
        np.cos(2*np.pi*dow/7.0),
    ]).T
    return TimeSeries.from_times_and_values(idx, X, columns=["dow_sin","dow_cos"])

def fourier_from_index(idx) -> TimeSeries:
    hour = idx.hour.to_numpy()
    dow  = idx.dayofweek.to_numpy()
    X = np.vstack([
        np.sin(2*np.pi*hour/24.0),
        np.cos(2*np.pi*hour/24.0),
        np.sin(2*np.pi*dow/7.0),
        np.cos(2*np.pi*dow/7.0),
    ]).T
    return TimeSeries.from_times_and_values(idx, X, columns=["h_sin","h_cos","dow_sin","dow_cos"])

def dim_indicator_norm(idx, D: int) -> TimeSeries:
    v = (np.arange(D, dtype=np.float32) / (D-1)).astype(np.float32)  # 0..1
    X = np.tile(v, (len(idx), 1))
    cols = [f"dim_id_{i}" for i in range(D)]
    return TimeSeries.from_times_and_values(idx, X, columns=cols)

def build_past_covs_552(ts_sc: TimeSeries, lags=(1,24,168), fourier_func=fourier_from_index_min) -> TimeSeries:
    lag_covs = lag_covs_from_scaled_target(ts_sc, lags)
    idx = lag_covs.time_index
    time_covs = fourier_func(idx)
    dim_covs  = dim_indicator_norm(idx, ts_sc.width)
    return concatenate([lag_covs, dim_covs, time_covs], axis=1)

def ts_upto(ts, end_time):
    if hasattr(ts, "slice_end"):
        return ts.slice_end(end_time)
    if hasattr(ts, "drop_after"):
        return ts.drop_after(end_time)
    if hasattr(ts, "split_after"):
        return ts.split_after(end_time)[0]
    return ts.slice(ts.start_time(), end_time)

def gluon_to_wide_df(dataset):
    series_list = []
    
    for i, entry in enumerate(dataset):
        idx = pd.date_range(
            start=entry["start"].to_timestamp(), 
            periods=len(entry["target"]), 
            freq=entry["start"].freqstr
        )
        series = pd.Series(entry["target"], index=idx, name=f"node_{i}")
        series_list.append(series)
    
    return pd.concat(series_list, axis=1)

def get_7_test_windows(dataset, num_nodes=137):
    all_series = []
    
    for entry in dataset:
        idx = pd.date_range(
            start=entry["start"].to_timestamp(), 
            periods=len(entry["target"]), 
            freq=entry["start"].freqstr
        )
        all_series.append(pd.Series(entry["target"], index=idx))
    
    num_windows = len(all_series) // num_nodes
    windows = []
    
    for w in range(num_windows):
        start_idx = w * num_nodes
        end_idx = (w + 1) * num_nodes
        window_df = pd.concat(all_series[start_idx:end_idx], axis=1)
        window_df.columns = [f"node_{i}" for i in range(num_nodes)]
        windows.append(window_df)
        
    return windows, all_series

def load_and_prepare_data(dataset_name: str, 
                          lags: Tuple, 
                          fourier_func_dict: dict):
    
    ds = get_dataset(dataset_name, regenerate=False)
    freq = ds.metadata.freq
    offset = to_offset(freq)
    granularity = norm_freq_str(offset.name) 
    
    train_list = list(ds.train)
    test_list  = list(ds.test)
    num_test_dates = len(test_list) // len(train_list)  
    
    target_dim = int(ds.metadata.feat_static_cat[0].cardinality) 
    train_grouper = MultivariateGrouper(max_target_dim=target_dim)
    test_grouper  = MultivariateGrouper(num_test_dates=num_test_dates, max_target_dim=target_dim)
    
    train_mv_items = list(train_grouper(train_list))   
    
    if dataset_name == 'kdd_cup_2018_without_missing':  
        for i in range(len(test_list)):
            if len(test_list[i]['target']) == 10898:
                test_list[i]['target'] = np.concatenate(
                    (test_list[i]['target'], np.zeros(8)), axis=0)
        dataset_test = test_grouper(test_list)
    else:
        dataset_test = test_grouper(test_list)

    test_mv_items  = list(dataset_test)     
    
    train_ts = gluonts_item_to_darts_mv(train_mv_items[0], freq)
    test_ts_list = [gluonts_item_to_darts_mv(it, freq) for it in test_mv_items]  

    y_scaler = Scaler()                     
    train_y_sc = y_scaler.fit_transform(train_ts)

    test_windows, all_series = get_7_test_windows(test_list, num_nodes=target_dim)
    
    fourier_func = fourier_func_dict.get(dataset_name, fourier_from_index)
    train_pc = build_past_covs_552(train_y_sc, lags=lags, fourier_func=fourier_func)
    
    train_y_sc = train_y_sc.slice_intersect(train_pc)
    train_pc   = train_pc.slice_intersect(train_y_sc)

    return {
        'scaler': y_scaler, 'train_ts': train_ts, 'train_y_sc': train_y_sc, 
        'test_ts_list': test_ts_list, 'test_windows': test_windows, 
        'all_series': all_series, 'train_pc': train_pc, 'freq': granularity,
        'fourier_func': fourier_func
    }

FOURIER_DICT = {
    'solar_nips': fourier_from_index, 
    'wiki2000_nips': fourier_from_index_day, 
    'electricity_nips': fourier_from_index, 
    'taxi_30min': fourier_from_index_min,
    'kdd_cup_2018_without_missing': fourier_from_index, 
    'traffic_nips': fourier_from_index
}
