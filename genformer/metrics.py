import numpy as np
import pandas as pd
import torch
from gluonts.model.forecast import SampleForecast
from gluonts.evaluation import MultivariateEvaluator
from darts import TimeSeries

from genformer.data import ts_upto, build_past_covs_552

def to_gluonts_multivariate_inputs(
    preds: np.ndarray,
    targets: np.ndarray,
    start_dates,
    freq: str = "H",
    item_ids=None,
    past_targets: np.ndarray | None = None,
    dtype=np.float32,
):
    preds = np.asarray(preds, dtype=dtype)
    targets = np.asarray(targets, dtype=dtype)

    B, N, T, D = preds.shape

    if isinstance(start_dates, (str, pd.Timestamp, pd.Period)) or not hasattr(start_dates, "__len__"):
        start_dates = [start_dates] * B

    if item_ids is None:
        item_ids = [f"item_{i}" for i in range(B)]

    targets_list = []
    forecasts_list = []

    columns = list(range(D))  

    for i in range(B):
        start_period = pd.Period(start_dates[i], freq=freq)
        forecast = SampleForecast(
            samples=preds[i],          
            start_date=start_period,
            item_id=item_ids[i],
        )
        forecasts_list.append(forecast)

        if past_targets is None:
            target_index = pd.period_range(start=start_period, periods=T, freq=freq)
            target_values = targets[i]  
        else:
            H = past_targets.shape[1]
            target_index = pd.period_range(start=start_period - H, periods=H + T, freq=freq)
            target_values = np.concatenate([past_targets[i], targets[i]], axis=0)  

        target_df = pd.DataFrame(target_values, index=target_index, columns=columns)
        targets_list.append(target_df)

    return targets_list, forecasts_list

def crps(preds, targets, quantiles=(np.arange(20) / 20.0)[1:]):
    x = np.quantile(preds, quantiles, axis=1, method="nearest")  
    quantiles = np.expand_dims(quantiles, axis=list(range(1, len(preds.shape))))  
    loss = 2 * np.sum(np.abs((x - targets) * ((targets <= x) - quantiles)), axis=2)  
    return loss.mean() / np.abs(targets).sum(axis=1).mean()

def crps_sum(preds, targets, quantiles=(np.arange(20) / 20.0)[1:], frequency='D'):
    preds_sum = preds.sum(axis=-1)      
    targets_sum = targets.sum(axis=-1)  
    return crps(preds_sum, targets_sum, quantiles=quantiles)

def get_metric_and_prediction(model, test_windows, y_scaler, pred_len=24, lags=(1, 24, 168), num_samples=100, seed=42, std=None, fourier_func=None, is_clip=False, frequency='D'):
    all_forecasts = []
    all_targets = []
    to_save_forecast = []
    to_save_targets = []

    for i, window_df in enumerate(test_windows):
        full_ts = TimeSeries.from_dataframe(window_df).astype(np.float32)
        full_sc = y_scaler.transform(full_ts).astype(np.float32)
        full_pc = build_past_covs_552(full_sc, lags=lags, fourier_func=fourier_func).astype(np.float32)

        full_sc = full_sc.slice_intersect(full_pc)
        full_ts = full_ts.slice_intersect(full_sc)
        full_pc = full_pc.slice_intersect(full_sc)

        past_true_sc = full_sc[:-pred_len]
        gt_future    = full_ts[-pred_len:]
        
        forecast_start = gt_future.start_time()
        pc_past = ts_upto(full_pc, forecast_start)

        model.model.encoder[0].reset_seed(seed)
        if std is not None:
            model.model.encoder[0].reset_std(std)
        
        fc_sc = model.predict(
            n=pred_len,
            series=past_true_sc,
            past_covariates=pc_past,
            num_samples=num_samples,
            verbose=False,
            random_state=1456445
        )

        fc = y_scaler.inverse_transform(fc_sc)
        if is_clip:
            fc = fc.with_values(np.clip(fc.all_values(), a_min=0, a_max=None))

        to_save_forecast.append(fc)
        to_save_targets.append(gt_future)
        
        all_forecasts.append(fc.all_values(copy=False))
        all_targets.append(gt_future.all_values(copy=False))

    stacked_forecasts = np.stack(all_forecasts, axis=0)      
    preds_reshaped = np.transpose(stacked_forecasts, (0, 3, 1, 2))  

    stacked_targets = np.stack(all_targets, axis=0)          
    targets_reshaped = np.squeeze(stacked_targets, axis=-1)  

    target_list , forecast_list = to_gluonts_multivariate_inputs(preds_reshaped, targets_reshaped, pd.Timestamp(gt_future.start_time()), freq=frequency)

    evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum})
    agg_metric, item_metrics = evaluator(target_list, forecast_list)

    print(f"======= Evaluation metrics for models =======")
    print("CRPS:", agg_metric["mean_wQuantileLoss"])
    print("ND:", agg_metric["ND"])
    print("NRMSE:", agg_metric["NRMSE"])
    print("")
    print("CRPS-Sum:", agg_metric["m_sum_mean_wQuantileLoss"])
    print("ND-Sum:", agg_metric["m_sum_ND"])
    print("NRMSE-Sum:", agg_metric["m_sum_NRMSE"])
    crps_ours = crps_sum(preds_reshaped, targets_reshaped)
    print("CRPS Ours:", crps_ours)
    
    return agg_metric["m_sum_mean_wQuantileLoss"], agg_metric["m_sum_ND"], agg_metric["m_sum_NRMSE"], crps_ours, to_save_forecast, to_save_targets

def energy_score_loss(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    M = samples.size(0)
    dist_to_target = torch.linalg.norm(samples - target.unsqueeze(0), dim=-1).mean(0)
    s = samples.reshape(M, -1, samples.size(-1))
    diff = s.unsqueeze(1) - s.unsqueeze(0)
    pairwise_dist = torch.linalg.norm(diff, dim=-1).mean(dim=(0, 1))
    dist_samples = pairwise_dist.view(target.size(0), target.size(1))
    loss = dist_to_target - 0.5 * dist_samples
    return loss.mean()
