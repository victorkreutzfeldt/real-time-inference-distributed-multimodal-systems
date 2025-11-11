#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

"""
Plot Average Hamming Accuracy Comparison for Different Wrappers at Specified Audio SNR.

This script loads precomputed results from multiple wrapper variants evaluated at a given audio SNR value obtained 
by running run_wrapper.py.
It computes aggregate statistics (averages and standard deviations) for performance metrics like Hamming accuracy
across temporal windows, handling variable sequence lengths with NaN padding.

It generates and saves:
    - Comparative scatter plots of average Hamming accuracy versus inference time.
    - TikZ coordinate files for seamless integration of plots in LaTeX documents.

Arguments:
    --audio_snr_dB : str
        Audio signal-to-noise ratio in decibels; must be one of "-5", "1.1888", or "2".

Outputs:
    - PNG plot images saved under 'data/results/figs/'.
    - TikZ coordinate files saved under 'data/results/tikz/'.

Usage:
    python plot_accuracy_comparison.py --audio_snr_dB <SNR_value>
    e.g.:
    python plot_accuracy_comparison.py --audio_snr_dB 1.1888

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

import argparse


parser = argparse.ArgumentParser(
    description="Plot for a given SNR in dB the comparison between the baseline and neuro-inspired wrappers."
)

parser.add_argument(
    "--audio_snr_dB",
    type=str,
    required=True,
    choices=["-5", "1.1888", "2"],
    help="Define the SNR for audio in decibels. Choices are: -5, 1.1888, 2 [dB]."
)

args = parser.parse_args()


def compute_statistics(results: dict) -> tuple[dict, int]:
    """
    Compute average and std metrics across videos, padding sequences to the maximum length with NaNs,
    and using nan-aware aggregation functions.

    Args:
        results (dict): Dictionary containing per-video metrics including nested modality dictionaries.

    Returns:
        tuple:
            - dict: Aggregated statistics with 'avg_' and 'std_' prefixed metrics.
            - int: Maximum number of temporal windows of integration (TWIs) across videos.

    Notes:
        Pads shorter sequences with NaNs instead of repeating last value to avoid biasing stats.
        Applies np.nanmean and np.nanstd to ignore NaNs during aggregation.
    """

    max_num_twis = max(results['num_twis'])

    def pad_with_nans(arrays, target_len):
        """
        Pad list of 1D arrays (or lists) with NaNs to target_len along axis=0.

        Parameters:
            arrays (list of arrays): input arrays per video.
            target_len (int): target sequence length to pad to.

        Returns:
            np.ndarray: 2D array with shape (num_videos, target_len).
        """
        padded = []
        for arr in arrays:
            arr = np.array(arr)
            current_len = arr.shape[0]

            if current_len == target_len:
                padded.append(arr)
            elif current_len < target_len:
                nan_padding = np.full((target_len - current_len,), np.nan)
                padded.append(np.concatenate([arr, nan_padding]))
            else:
                padded.append(arr[:target_len])
        return np.stack(padded)

    # Pad and aggregate scalar per-video lists
    hamming_acc = pad_with_nans(results['hamming_acc'], max_num_twis)
    subset_acc = pad_with_nans(results['subset_acc'], max_num_twis)

    # Handle nested modality dictionaries
    def pad_and_aggregate_dict(data_dict):
        agg = {}
        for modality, data_list in data_dict.items():
            padded_data = pad_with_nans(data_list, max_num_twis)
            agg[modality] = {
                'avg': np.nanmean(padded_data, axis=0),
                #'std': np.nanstd(padded_data, axis=0, ddof=1)
            }
        return agg

    num_rx_pkts_stats = pad_and_aggregate_dict(results['num_rx_pkts'])
    curr_done_tokens_stats = pad_and_aggregate_dict(results['curr_num_done_tokens'])

    # Aggregate num_missed_pkts and num_twis: ignoring None values
    def nanmean_ignore_none(lst):
        arr = np.array([x if x is not None else np.nan for x in lst], dtype=float)
        return np.nanmean(arr)

    avg_num_missed_pkts = {
        modality: nanmean_ignore_none(results['num_missed_pkts'][modality])
        for modality in results['num_missed_pkts']
    }
    avg_num_twis = nanmean_ignore_none(results['num_twis'])

    stats = {
        'hamming_acc': {
            'avg': np.nanmean(hamming_acc, axis=0),
            'std': np.nanstd(hamming_acc, axis=0, ddof=1)
        },

        'subset_acc': {
            'avg': np.nanmean(subset_acc, axis=0),
            'std': np.nanstd(subset_acc, axis=0, ddof=1)
        },

        'num_rx_pkts': num_rx_pkts_stats,
        'curr_num_done_tokens': curr_done_tokens_stats,

        'avg_num_missed_pkts': avg_num_missed_pkts,
        'avg_num_twis': avg_num_twis
    }

    return stats, max_num_twis

# ---- Main -----
def main() -> None:
    """
    Main plotting function to compare accuracy performance across wrapper variants for a given SNR.

    Loads results from compressed pickle files for variants 'SotA', 'PaMo', and 'ToMo' and
    plots the average Hamming accuracy across inference windows over time. Saves figures
    and TikZ format plot coordinates for LaTeX integration.

    Side Effects:
        Creates 'data/results/figs/' and 'data/results/tikz/' directories if they do not exist.
        Saves PNG plot images and TikZ coordinate files corresponding to variants and SNR.
    """
    # Save figure path and TikZ coordinates folder
    SAVE_FIG = 'data/results/figs/'
    TIKZ_COORDS_PATH = 'data/results/tikz/'

    os.makedirs(SAVE_FIG, exist_ok=True)
    os.makedirs(TIKZ_COORDS_PATH, exist_ok=True)

    # Variants
    variants = ['SotA', 'PaMo', 'ToMo'] 

    plt.figure(figsize=(9, 5))

    # Go over variants
    for variant in variants:
        print(f"Processing variant: {variant}")

        # Load the results dictionary
        results_path = f'data/results/{variant}_SNR_{float(args.audio_snr_dB)}.pkl.gz' 
        try:
            with gzip.open(results_path, 'rb') as f:
                results = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {results_path}.")
            exit(1)
        
        # Extract window duration and stop time
        TWI_DURATION = results['metadata']['twi_duration']
        STOP_TIME = results['metadata']['stop_time']
    
        # Compute statistics
        stats, max_num_twis = compute_statistics(results)

        # Define axes in seconds
        if variant == 'SotA':
            x_data = max_num_twis * TWI_DURATION
            y_data = float(stats['hamming_acc']['avg'].item())
        else:
            twis = np.arange(1, max_num_twis + 1)
            x_data = twis * TWI_DURATION
            y_data = stats['hamming_acc']['avg']
            
        # Plot average hamming accuracy vs time
        label = rf"{variant}"

        if variant == 'PaMo':
            marker_style = 'o'
        else:
            marker_style = 'd'

        if variant == 'SotA':
            plt.scatter(x_data, y_data, marker='s', s=50, label=label)
        else:   
            plt.scatter(x_data[:-1], y_data[:-1], marker=marker_style, s=2, label=label)

        # Convert coordinates to TikZ format suitable for \addplot coordinates {...}
        if variant == 'SotA':
            tikz_coords = f"({x_data:.4f},{y_data:.4f})"
        else:
            tikz_coords = " ".join([f"({x:.4f},{y:.4f})" for x, y in zip(x_data, y_data)]) 
        
        filename = os.path.join(TIKZ_COORDS_PATH, f"{variant}_SNR_{args.audio_snr_dB}.txt")
        with open(filename, "w") as f:
            f.write(tikz_coords)

        print(f"Saved TikZ coordinates to {filename}")

    plt.title(r'SNR = ' + f'{args.audio_snr_dB} dB', fontsize=14)

    plt.xlabel(r'Inference Time (s)')
    plt.ylabel(r'Average Hamming Accuracy')

    plt.grid(True, linewidth=0.5)
    
    plt.legend(fontsize=8, fancybox=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    
    plt.savefig(SAVE_FIG + f'accuracy_{variant}_SNR_{args.audio_snr_dB}.png', dpi=300)


if __name__ == "__main__":
    main()