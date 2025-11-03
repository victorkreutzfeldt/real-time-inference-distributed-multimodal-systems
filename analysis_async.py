import os
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Save figure path and TikZ coordinates folder
SAVE_FIG = 'data/communication/async/'
TIKZ_COORDS_PATH = 'data/communication/tikz_coords/'
os.makedirs(SAVE_FIG, exist_ok=True)
os.makedirs(TIKZ_COORDS_PATH, exist_ok=True)


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

        'avg_curr_num_completed_chunks_audio': np.mean(data['curr_num_completed_chunks_audio'], axis=0),
        'avg_curr_num_completed_chunks_video': np.mean(data['curr_num_completed_chunks_video'], axis=0),

        'std_hamming_acc':        np.std(data['hamming_acc'], axis=0, ddof=1),
        'std_subset_acc':         np.std(data['subset_acc'], axis=0, ddof=1),
    }
    
    return stats


if __name__ == "__main__":

    # Specify your target SNR (e.g., 0)
    target_snr = 2

    # Load the results dictionary
    async_results_path = f'data/communication/async/pomo_per_video_aligned_deterministic_SNR_{target_snr}.gz' 
    with gzip.open(async_results_path, 'rb') as f:
        data = pickle.load(f)

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

    # Plot average hamming accuracy vs time
    plt.figure(figsize=(9, 5))
    label = rf"Window Duration: $max(\bar{{T}}_a, \bar{{T}}_v) = {WINDOW_DURATION:.2f} \mathrm{{s}}$"
    plt.plot(x_data, y_data, linestyle='-', label=label, color='blue')

    plt.xlabel(r'Inference Time (s)')
    plt.ylabel(r'Average Hamming Accuracy')
    plt.grid(True, linewidth=0.5)
    plt.legend(fontsize=8, fancybox=True, shadow=True, borderpad=1)
    plt.tight_layout()
    plt.savefig(SAVE_FIG + f'accuracy_SNR_{target_snr}.png', dpi=300)
    # plt.show()

    # --- Save TikZ coordinates for avg_hamming_acc at current SNR ---

    # Convert coordinates to TikZ format suitable for \addplot coordinates {...}
    tikz_coords = " ".join([f"({x:.6f},{y:.6f})" for x, y in zip(x_data, y_data)])

    filename = os.path.join(TIKZ_COORDS_PATH, f"pomo_avg_hamming_acc_snr_{target_snr}.txt")
    with open(filename, "w") as f:
        f.write(tikz_coords)

    print(f"Saved TikZ coordinates to {filename}")



# import os
# import gzip
# import pickle
# import numpy as np

# import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

# # Save figure path
# SAVE_FIG = 'data/communication/async/'
# os.makedirs(SAVE_FIG, exist_ok=True)


# def compute_statistics(data):
#     """
    
#     """

#     # Check if all videos were processed after the same amount of windows
#     num_windows_list = data['metadata']['NUM_WINDOWS']
#     assert all(x == num_windows_list[0] for x in num_windows_list), "All videos must have the same number of windows processed."

#     stats = {
#         'avg_hamming_acc':        np.mean(data['hamming_acc'], axis=0),
#         'avg_subset_acc':         np.mean(data['subset_acc'], axis=0),

#         'avg_num_rx_pkts_audio':  np.mean(data['num_rx_pkts_audio'], axis=0),
#         'avg_num_rx_pkts_video':  np.mean(data['num_rx_pkts_video'], axis=0),

#         'avg_curr_num_completed_chunks_audio': np.mean(data['curr_num_completed_chunks_audio'], axis=0),
#         'avg_curr_num_completed_chunks_video': np.mean(data['curr_num_completed_chunks_video'], axis=0),

#         'std_hamming_acc':        np.std(data['hamming_acc'], axis=0, ddof=1),
#         'std_subset_acc':         np.std(data['subset_acc'], axis=0, ddof=1),
        
#         #'std_causal_hamming_acc': data['causal_hamming_acc'].std(axis=0, ddof=1),
#         #'std_causal_subset_acc':  data['causal_subset_acc'].std(axis=0, ddof=1),
#         #'std_num_rx_pkts_audio':  data['num_rx_pkts_audio'].std(axis=0, ddof=1),
#         #'std_num_rx_pkts_video':  data['num_rx_pkts_video'].std(axis=0, ddof=1),
#         #'std_curr_samples_audio': data['curr_samples_audio'].std(axis=0, ddof=1),
#         #'std_curr_samples_video': data['curr_samples_video'].std(axis=0, ddof=1)
#     }
    
#     return stats


# # def compute_statistics_per_chunk(data):
# #     stats = {
# #         'avg_hamming_acc':        data['hamming_acc'].mean(axis=0),
# #         'avg_subset_acc':         data['subset_acc'].mean(axis=0),
# #         'avg_num_rx_pkts_audio':  data['num_rx_pkts_audio'].mean(axis=0),
# #         'avg_num_rx_pkts_video':  data['num_rx_pkts_video'].mean(axis=0),
# #         'avg_curr_samples_audio': data['curr_samples_audio'].mean(axis=0),
# #         'avg_curr_samples_video': data['curr_samples_video'].mean(axis=0),

# #         'std_hamming_acc':        data['hamming_acc'].std(axis=0, ddof=1),
# #         'std_subset_acc':         data['subset_acc'].std(axis=0, ddof=1),
# #         'std_num_rx_pkts_audio':  data['num_rx_pkts_audio'].std(axis=0, ddof=1),
# #         'std_num_rx_pkts_video':  data['num_rx_pkts_video'].std(axis=0, ddof=1),
# #         'std_curr_samples_audio': data['curr_samples_audio'].std(axis=0, ddof=1),
# #         'std_curr_samples_video': data['curr_samples_video'].std(axis=0, ddof=1)
# #     }

# #     return stats


# if __name__ == "__main__":

#     # Load the results dictionaries
#     async_results_path = 'data/communication/async/per_video_aligned_deterministic_SNR_0.gz' 
#     with gzip.open(async_results_path, 'rb') as f:
#         data = pickle.load(f)

#     # Get window duration from metadata
#     WINDOW_DURATION = data['metadata']['WINDOW_DURATION']
    
#     # Compute statsistics
#     stats = compute_statistics(data)

#     # Define time axis in seconds
#     num_windows = data['metadata']['NUM_WINDOWS'][0]  # All videos have the same number of windows
#     windows = np.arange(num_windows)
#     t_seconds = windows * WINDOW_DURATION 
#     t_seconds = np.array(t_seconds, dtype=float)
    
#     # asynch_results_path = 'data/asynch/per_video_aligned_mode_no-discard_lp_0.0.gz' 
#     # with gzip.open(asynch_results_path, 'rb') as f:
#     #     data_no_discard = pickle.load(f)

#     # asynch_results_path_per_chunk = 'data/asynch/per_chunk_aligned_mode_past_lp_0.0.gz'
#     # with gzip.open(asynch_results_path_per_chunk, 'rb') as f:
#     #     data_per_chunk_past = pickle.load(f)  

#     # asynch_results_path_per_chunk = 'data/asynch/per_chunk_aligned_mode_no-discard_lp_0.0.gz'
#     # with gzip.open(asynch_results_path_per_chunk, 'rb') as f:
#     #     data_per_chunk_no_discard = pickle.load(f)  

#     # asynch_results_drifted = 'data/asynch/per_video_drifted_mode_past_lp_0.0_drift_4.gz'
#     # with gzip.open(asynch_results_drifted, 'rb') as f:
#     #     data_drifted_past_4 = pickle.load(f)  

#     # asynch_results_drifted = 'data/asynch/per_video_drifted_mode_no-discard_lp_0.0_drift_4.gz'
#     # with gzip.open(asynch_results_drifted, 'rb') as f:
#     #     data_drifted_no_discard_4 = pickle.load(f) 

#     # asynch_results_drifted = 'data/asynch/per_video_drifted_mode_past_lp_0.0_drift_8.gz'
#     # with gzip.open(asynch_results_drifted, 'rb') as f:
#     #     data_drifted_past_8 = pickle.load(f) 

#     # asynch_results_drifted = 'data/asynch/per_video_drifted_mode_no-discard_lp_0.0_drift_8.gz'
#     # with gzip.open(asynch_results_drifted, 'rb') as f:
#     #     data_drifted_no_discard_8 = pickle.load(f) 


#     # # Extract the number of windows
#     # num_windows = data['hamming_acc'].shape[1]
#     # windows = np.arange(num_windows)
#     # t_seconds = windows * WINDOW_DURATION    # time axis in seconds

#     # # Compute statistics
#     # stats_past = compute_statistics_per_video(data)
#     # stats_no_discard = compute_statistics_per_video(data_no_discard)

#     # stats_per_chunk_past = compute_statistics_per_chunk(data_per_chunk_past)
#     # stats_per_chunk_no_discard = compute_statistics_per_chunk(data_per_chunk_no_discard)

#     # stats_past_drifted_4 = compute_statistics_per_video(data_drifted_past_4)
#     # stats_no_discard_drifted_4 = compute_statistics_per_video(data_drifted_no_discard_4)

#     # stats_past_drifted_8 = compute_statistics_per_video(data_drifted_past_8)
#     # stats_no_discard_drifted_8 = compute_statistics_per_video(data_drifted_no_discard_8)

#     # --- 1. Plot Average Accuracies vs Window Index/Time ---
#     plt.figure(figsize=(9, 5))

#     # Plot lines with LaTeX formatted labels
#     # plt.plot(t_seconds, stats_past['avg_hamming_acc'], linestyle='-', label=r"Per video inference | mode: 'past'")
#     # plt.plot(t_seconds, stats_no_discard['avg_hamming_acc'], linestyle='-', label=r"Per video inference | mode: 'no-discard'")

#     label = rf"Window Duration: $max(\bar{{T}}_a, \bar{{T}}_v) = {WINDOW_DURATION:.2f} \mathrm{{s}}$"
#     breakpoint()
#     plt.plot(t_seconds, stats['avg_hamming_acc'][:num_windows], linestyle='-', label=label, color='blue')

#     #plt.plot(t_seconds, stats_no_discard['avg_causal_hamming_acc'], linestyle='-', label=r"Per video inference | mode: 'no-discard' | causal")

#     #plt.plot(t_seconds, stats_per_chunk_past['avg_hamming_acc'], linestyle='--', label=r"Per chunk inference | mode: 'past'")
#     #plt.plot(t_seconds, stats_per_chunk_no_discard['avg_hamming_acc'], linestyle='--', label=r"Per chunk inference | mode: 'no-discard'")

#     #plt.plot(t_seconds, stats_past_drifted_4['avg_causal_hamming_acc'], linestyle='-.', label=r"Per video inference | mode: 'past' | drift = 4")
#     #plt.plot(t_seconds, stats_no_discard_drifted_4['avg_causal_hamming_acc'], linestyle='-.', label=r"Per video inference | mode: 'no-discard' | drift = 4")

#     #plt.plot(t_seconds, stats_past_drifted_8['avg_causal_hamming_acc'], linestyle=':', label=r"Per video inference | mode: 'past' | drift = 8")
#     #plt.plot(t_seconds, stats_no_discard_drifted_8['avg_causal_hamming_acc'], linestyle=':', label=r"Per video inference | mode: 'no-discard' | drift = 8")

#     # plt.plot(t_seconds, stats['avg_subset_acc'], marker='s', label=r'Avg. Subset Accuracy', color='g')

#     # Uncomment if you want to add shaded std deviation areas
#     #a = np.array(stats['avg_hamming_acc'], dtype=float)
#     #s = np.array(stats['std_hamming_acc'], dtype=float)
#     #plt.fill_between(t_seconds, a - s, a + s, color='b', alpha=0.15)

#     # plt.fill_between(t_seconds,
#     #                  stats['avg_causal_hamming_acc'] - stats['std_causal_hamming_acc'],
#     #                  stats['avg_causal_hamming_acc'] + stats['std_causal_hamming_acc'],
#     #                  color='g', alpha=0.15)

#     # plt.fill_between(t_seconds,
#     #                  stats['avg_subset_acc'] - stats['std_subset_acc'],
#     #                  stats['avg_subset_acc'] + stats['std_subset_acc'],
#     #                  color='g', alpha=0.15)

#     plt.xlabel(r'Inference Time (s)')
#     plt.ylabel(r'Average Hamming Accuracy')
#     #plt.title(r'Average Hamming Accurac vs Time')

#     plt.grid(True, linewidth=0.5)

#     # Improved legend with smaller font, rounded box, shadow, and padding
#     plt.legend(fontsize=8, fancybox=True, shadow=True, borderpad=1)

#     plt.tight_layout()

#     plt.savefig(SAVE_FIG + 'accuracy.png', dpi=300)

#     #plt.show()
#     #breakpoint()
    
#     # # --- 2. Plot Average Number of Received Audio Packets per Window ---
#     # plt.figure(figsize=(9, 5))

#     # plt.scatter(t_seconds, stats['avg_num_rx_pkts_audio'], marker='o', s=20, color='b', label='Audio')
#     # plt.scatter(t_seconds, stats['avg_num_rx_pkts_video'], marker='x', s=20, color='b', label='Video')
     

#     # # plt.fill_between(t_seconds,
#     # #                  stats_past['avg_num_rx_pkts_audio'] - stats_past['std_num_rx_pkts_audio'],
#     # #                  stats_past['avg_num_rx_pkts_audio'] + stats_past['std_num_rx_pkts_audio'],
#     # #                  color='orange', alpha=0.15)
    
#     # plt.xlabel(r'Inference Time (s)')
#     # plt.ylabel(r'Average Number of Received Packets')
    
#     # #plt.title('Average RX Audio Packets per Window')
#     # plt.grid(True, linewidth=0.5)
    
#     # plt.legend(fontsize=8, fancybox=True, shadow=True, borderpad=1)
    
#     # plt.tight_layout()

#     # plt.savefig(SAVE_FIG + 'rx_packets.png', dpi=300)

    
#     # # --- 3. Plot Average Current Audio Samples per Window ---
#     # plt.figure(figsize=(9, 5))

#     # plt.scatter(t_seconds, stats['avg_curr_num_completed_chunks_audio'], marker='o', s=20, color='b', label='Audio')
#     # plt.scatter(t_seconds, stats['avg_curr_num_completed_chunks_video'], marker='x', s=20, color='b', label='Video')

#     # # plt.fill_between(t_seconds,
#     # #                  stats_past['avg_curr_samples_audio'] - stats_past['std_curr_samples_audio'],
#     # #                  stats_past['avg_curr_samples_audio'] + stats_past['std_curr_samples_audio'],
#     # #                  color='green', alpha=0.15)
    
#     # plt.xlabel(r'Inference Time (s)')
#     # plt.ylabel(r'Average Cumulative Number of Completed Chunks')
    
#     # #plt.title('Average Current Audio Samples per Window')
    
#     # plt.grid(True, linewidth=0.5)
    
#     # plt.legend()
    
#     # plt.tight_layout()
    
#     # plt.savefig(SAVE_FIG + 'curr_num_completed_chunks.png', dpi=300)

#     # breakpoint()