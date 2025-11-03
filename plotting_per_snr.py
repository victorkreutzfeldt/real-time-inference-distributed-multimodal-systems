import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

# Save figure path and TikZ coordinates folder
SAVE_FIG = 'data/results/figs/'
TIKZ_COORDS_PATH = 'data/results/tikz/'

os.makedirs(SAVE_FIG, exist_ok=True)
os.makedirs(TIKZ_COORDS_PATH, exist_ok=True)

import argparse

parser = argparse.ArgumentParser(
    description="Plot for a given SNR in dB the comparison between the baseline and neuro-inspired wrappers."
)

parser.add_argument(
    "--snr_dB",
    type=str,
    required=True,
    choices=["-5", "1.1888", "2"],
    help="Define the SNR for audio in decibels. Choices are: -5, 1.1888, 2 (dB)."
)

args = parser.parse_args()

def compute_statistics(data):
    """
    Compute average and std metrics across videos.
    """

    # Check all videos processed same number of windows
    num_windows_list = data['metadata']['NUM_WINDOWS']
    assert all(x == num_windows_list[0] for x in num_windows_list), "All videos must have the same number of windows processed."

    stats = {
        'avg_hamming_acc':        np.mean(data['hamming_acc'], axis=0),
        'avg_subset_acc':         np.mean(data['subset_acc'], axis=0),

        'avg_num_rx_pkts_audio':  np.mean(data['num_rx_pkts_audio'], axis=0),
        'avg_num_rx_pkts_video':  np.mean(data['num_rx_pkts_video'], axis=0),

        'avg_curr_num_completed_tokens_audio': np.mean(data['curr_num_completed_tokens_audio'], axis=0),
        'avg_curr_num_completed_tokens_video': np.mean(data['curr_num_completed_tokens_video'], axis=0),

        'std_hamming_acc':        np.std(data['hamming_acc'], axis=0, ddof=1),
        'std_subset_acc':         np.std(data['subset_acc'], axis=0, ddof=1),
    }
    
    return stats


if __name__ == "__main__":

    # Variants
    variants = ['PaMo', 'ToMo'] 

    plt.figure(figsize=(9, 5))

    # Go over variants
    for variant in variants:

        print(f"Processing variant: {variant}")

        # Load the results dictionary
        results_path = f'data/results/{variant}_SNR_{args.snr_dB}.gz' 

        try:
            with gzip.open(results_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {results_path}.")
            exit(1)

        WINDOW_DURATION = data['metadata']['WINDOW_DURATION']
        num_windows = data['metadata']['NUM_WINDOWS'][0]
        
        # Compute statistics
        stats = compute_statistics(data)

        # Define time axis in seconds
        windows = np.arange(1, num_windows+1)
        t_seconds = windows * WINDOW_DURATION 
        t_seconds = np.array(t_seconds, dtype=float)

        x_data = t_seconds
        y_data = stats['avg_hamming_acc'][:num_windows]

        if variant == 'PaMo':
            sota_pamo = (x_data[-1], y_data[-1])
        else:
            sota_tomo = (x_data[-1], y_data[-1])

        # Plot average hamming accuracy vs time
        
        label = rf"{variant}"

        if variant == 'PaMo':
            marker_style = 'o'
        else:
            marker_style = 'd'

        plt.scatter(x_data[:-1], y_data[:-1], marker=marker_style, s=2, label=label)

        # Convert coordinates to TikZ format suitable for \addplot coordinates {...}
        tikz_coords = " ".join([f"({x:.6f},{y:.6f})" for x, y in zip(x_data, y_data)])
        filename = os.path.join(TIKZ_COORDS_PATH, f"{variant}_SNR_{args.snr_dB}.txt")
        
        with open(filename, "w") as f:
            f.write(tikz_coords)

        print(f"Saved TikZ coordinates to {filename}")

    print(sota_pamo, sota_tomo)
    
    plt.plot(sota_pamo[0], sota_pamo[1], 'ro', label='SotA')

    plt.title(r'SNR = ' + f'{args.snr_dB} dB', fontsize=14)

    plt.xlabel(r'Inference Time (s)')
    plt.ylabel(r'Average Hamming Accuracy')
    plt.grid(True, linewidth=0.5)
    
    plt.legend(fontsize=8, fancybox=True, shadow=True, borderpad=1)
    
    plt.tight_layout()
    
    plt.savefig(SAVE_FIG + f'accuracy_{variant}_SNR_{args.snr_dB}.png', dpi=300)
