import os
import argparse
import configparser
import time
import random
import numpy as np
import torch
import lightning as pl
import pandas as pd

from genformer.utils import deterministic, setup_logger
from genformer.data import load_and_prepare_data, FOURIER_DICT
from genformer.models import Enformer
from genformer.metrics import get_metric_and_prediction

def parse_tuple(val):
    return tuple(int(x.strip()) for x in val.split(','))

def main():
    parser = argparse.ArgumentParser(description="Run Enformer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. solar_nips")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config.ini")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save PyTorch Lightning model checkpoints")
    args = parser.parse_args()

    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config)
    if args.dataset not in config:
        raise ValueError(f"Dataset block [{args.dataset}] not found in config.")

    cfg = config[args.dataset]

    # Hyperparams string-parsing
    seed = cfg.getint('seed', 42)
    noise_std = cfg.getfloat('noise_std')
    num_samples = cfg.getint('num_samples_engression')
    d_model = cfg.getint('d_model')
    lr = cfg.getfloat('lr')
    n_coder_layers = cfg.getint('n_coder_layers')
    n_heads = cfg.getint('n_heads')
    d_ff = cfg.getint('d_ff')
    dropout = cfg.getfloat('dropout')
    activation = cfg.get('activation')
    batch_size = cfg.getint('batch_size')
    noise_dist = cfg.get('noise_dist')
    pred_len = cfg.getint('pred_len')
    is_clip = cfg.getboolean('is_clip')
    lags = parse_tuple(cfg.get('lags'))

    # Setup deterministic logic
    setup_logger()
    deterministic.init_all(seed)
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    print(f"Loading and processing {args.dataset}...")
    data_dict = load_and_prepare_data(
        dataset_name=args.dataset,
        lags=lags,
        fourier_func_dict=FOURIER_DICT
    )

    print("Initializing Model...")
    model = Enformer(
        input_chunk_length=pred_len,
        output_chunk_length=pred_len,
        num_samples_engression=num_samples,
        noise_std=noise_std,
        d_model=d_model,
        optimizer_kwargs={'lr': lr},
        output_chunk_shift=0,
        nhead=n_heads,
        num_encoder_layers=n_coder_layers,
        num_decoder_layers=n_coder_layers,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation=activation,
        batch_size=batch_size,
        n_epochs=n_epochs,
        noise_dist=noise_dist,
        random_state=seed,
        save_checkpoints=args.save_checkpoints
    )

    print("Training Model...")
    start_time = time.time()
    model.fit(
        data_dict['train_y_sc'],
        past_covariates=data_dict['train_pc'],
        verbose=True,
        dataloader_kwargs={"num_workers": 0}
    )
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    print("Evaluating Model...")
    crps_sum, nrmse_sum, _, crps_sum_our, forecasts, true_targets = get_metric_and_prediction(
        model,
        test_windows=data_dict['test_windows'],
        y_scaler=data_dict['scaler'],
        seed=seed,
        std=noise_std,
        lags=lags,
        fourier_func=data_dict['fourier_func'],
        is_clip=is_clip,
        pred_len=pred_len,
        frequency=data_dict['freq']
    )
    
    print("\n[Final Evaluation Metrics]")
    print(f"CRPS-Sum: {crps_sum}")
    print(f"NRMSE-Sum: {nrmse_sum}")


    # Save results to CSV
    results_file = "results.csv"
    
    new_result = pd.DataFrame([{
        "Dataset": args.dataset,
        "Seed": seed,
        "CRPS-Sum": crps_sum,
        "NRMSE-Sum": nrmse_sum,
        "CRPS-Ours": crps_sum_our
    }])
    
    if os.path.exists(results_file):
        existing_results = pd.read_csv(results_file)
        updated_results = pd.concat([existing_results, new_result], ignore_index=True)
    else:
        updated_results = new_result
        
    updated_results.to_csv(results_file, index=False)
    print(f"\nResults successfully appended to {results_file}")

if __name__ == "__main__":
    main()
