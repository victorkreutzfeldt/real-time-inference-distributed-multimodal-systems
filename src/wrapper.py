# src/utils_wrapper.py

"""
Wrapper class for real-time, non-blocking multimodal inference over asynchronous data streams.

This module implements a Wrapper that processes asynchronous audio and video packet streams inspired by 
Temporal Windows of Integration (TWIs). It handles tokenization of incoming packets, feature extraction 
using modality-specific pipelines, temporal alignment of embeddings, and multimodal inference using a 
given model.

Key features:
    - Supports multiple modalities with synchronized presentation timestamps (PTS).
    - Zero-imputes incomplete or missing tokens for robustness under communication delays/loss.
    - Tracks per-token processing status and accumulates accuracy metrics in real-time.
    - Supports incremental batch processing within time windows defined by the TWI model.

@author Victor Kreutzfeldt (@victorkreutzfelt or @victorcroisfelt)
@date 2025-11-11
"""

import torch
import numpy as np

from fractions import Fraction
from collections import deque

from src.vggish_input import waveform_to_examples

from src.utils import hamming_accuracy_from_label_lists, subset_accuracy_from_label_lists


class Wrapper:
    """
    Wrapper for non-blocking inference over asynchronous multimodal data streams neuro-inspired by
    temporal windows of integration (TWIs).

    This class processes asynchronous streams from multiple modalities by performing
    tokenization, feature extraction through modality-specific pipelines, temporal alignment
    of embeddings, and multimodal inference. It incorporates mechanisms to handle missing or
    incomplete packets with zero-imputation, providing resilience to communication uncertainties.

    Notes:
        - Assumes modality streams are synchronized, sharing comparable presentation timestamps (PTS).
        - Expects PTS values for each modality to start at zero and span the full duration of input.
        - PTS timebases correspond to each modality's sampling rate.
        - Missing or not-yet-received data is zero-imputed to maintain continuous temporal processing.
    """

    def __init__(self, config, model, pipelines, labels, received_streams, fallbacks, device) -> None:
        """
        Initialize the wrapper engine.

        Args:
            config (dict): Global and per-modality configuration settings.
            model (torch.nn.Module): Pre-loaded inference model instance for predictions.
            pipelines (dict[str, torch.nn.Module]): Feature extractors for each modality.
            labels (torch.Tensor): Ground-truth labels of shape (num_tokens, num_classes).
            received_streams (dict[str, collections.deque]): Mapping from modality names to deque buffers of received packets.
            fallbacks (dict[str, torch.Tensor]): Mapping from modality names to fallback embeddings used if data is missing.
            device (torch.device or str): Device to perform tensor computations on, e.g., 'cpu' or 'cuda'.

        Notes:
            Modality keys are strings representing each modality, e.g., 'audio', 'video'.
            Initializes internal state including token intervals and buffers for each modality.
        """
        self.config = config
        self.model = model
        self.pipelines = pipelines
        self.labels = labels
        self.received_streams = received_streams
        self.fallbacks = fallbacks
        self.device = device
        
        # Get number of tokens per video and token duration
        num_tokens = self.config['global']['num_tokens']
        token_duration = self.config['global']['token_duration']

        # Initialize control unit (CU) state
        self.state = {}
        self.state['twi_counter'] = 1
        self.state['token_intervals'] = [
                    (token_idx * token_duration,
                    (token_idx + 1) * token_duration)
                    for token_idx in range(num_tokens)
        ]

        for modality in config['modalities'].keys():
            self.state[modality] = {
                'packet_buffer': deque(),
                'token_buffer': {
                    token_idx: {
                        'samples': None,
                        'embedding': self.fallbacks[modality],
                        'start_time': token_idx * token_duration,
                        'end_time': (token_idx + 1) * token_duration,
                        'available': False, # indicate if token has any collected samples
                        'completed': False, # indicate if token is completed (received all packets)
                        'done': False  # indicate if token has been fully processed and fully received
                    } for token_idx in range(num_tokens)
                },
                'process': True,
                'num_done_tokens': 0
            }

        # Results storage
        self.results = {
            'hamming_acc': [],
            'subset_acc': [],
            'num_rx_pkts': {m: [] for m in config['modalities'].keys()},
            'curr_num_done_tokens': {m: [] for m in config['modalities'].keys()},
            'num_missed_pkts': {m: None for m in config['modalities'].keys()},
            'num_twis': None
        }


    def run_inference(self) -> dict:
        """
        Run real-time non-blocking inference over asynchronous data streams.

        The method iterates over time windows, processes packets for each modality,
        tokenizes samples, applies feature extraction pipelines, and performs
        inference with the combined embeddings. It tracks accuracy scores and stops
        when all tokens for all modalities are processed or the stop time is reached.

        Returns:
            dict: Results containing accuracy metrics, packet statistics, and processing info.
        """
        # Go over windows
        while True:

            # Get start and end times of this window
            t_start = (self.state['twi_counter'] - 1) * self.config['global']['twi_duration']
            t_end = t_start + self.config['global']['twi_duration'] 

            # Go over received streams modality-wise
            for modality in self.config['modalities'].keys():

                if not self.state[modality]['process']:
                    continue

                # Get received stream for this modality
                received_stream = self.received_streams[modality]
                
                # Get packet buffer for this modality
                packet_buffer = self.state[modality]['packet_buffer']

                # Fill packet buffer for this modality
                num_rx_pkts = 0
                while len(received_stream) > 0 and (t_start < received_stream[0].arrival_time <= t_end):
                    pkt = received_stream.popleft()
                    packet_buffer.append(pkt)
                    num_rx_pkts += 1
 
                # Save number of received packets in this window
                self.results['num_rx_pkts'][modality].append(num_rx_pkts)

                # Align packets in the packet buffer based on PTS
                self.state[modality]['packet_buffer'] = deque(sorted(packet_buffer, key=lambda p: p.pts))

                # Tokenization: update tokens from packets, zero-imputing incomplete tokens and updating token_buffer
                self.update_tokens_from_packet_buffer(modality)

                # Process available tokens
                self.streaming_pipeline(modality)

                # Clear packets from packet buffer related to completed tokens
                self.clear_packet_buffer(modality)

                # Update number of done tokens for this modality
                num_done_tokens = sum(
                    token['done'] for token in self.state[modality]['token_buffer'].values()
                )
                self.state[modality]['num_done_tokens'] = num_done_tokens
                self.results['curr_num_done_tokens'][modality].append(int(num_done_tokens))

            # Get embeddings of all tokens through all modalities for inference 
            embs = self.get_embeddings()
            
            # Perform inference
            with torch.no_grad():
                try:
                    logits = self.model(embs['audio'], embs['video'])
                except Exception as e:
                    raise RuntimeError("Model could not be run! FATAL") from e
        
            # Compute probabilities
            probas = torch.sigmoid(logits).squeeze(0)

            # Compute predictions
            preds = (probas > 0.5)

            # Transform into lists
            preds_list = [torch.nonzero(row).squeeze(1).tolist() for row in preds]
            labels_list = [torch.nonzero(row).squeeze(1).tolist() for row in self.labels]
        
            # Compute hamming and subset accuracies
            hamming_acc = hamming_accuracy_from_label_lists(labels_list, preds_list)
            subset_acc = subset_accuracy_from_label_lists(labels_list, preds_list)

            # Store accuracies
            self.results['hamming_acc'].append(float(hamming_acc))
            self.results['subset_acc'].append(float(subset_acc))

            
            ## Check stop criteria


            # Update processing flags per modality
            for modality in self.config['modalities'].keys():
                if self.state[modality]['num_done_tokens'] == self.config['global']['num_tokens']:
                    self.state[modality]['process'] = False

            # Break if all modalities are done processing
            if all(not self.state[mod]['process'] for mod in self.config['modalities'].keys()):
                break
 
            # Can we fit one more window run? If not, break if stop time reached
            if (t_end + self.config['global']['twi_duration'])>= self.config['global']['stop_time']:
                break
            
            # Update TWI index
            self.state['twi_counter'] += 1

        # After while loop exits
        self.results['num_twis'] = self.state['twi_counter']
        self.compute_num_missed_packets()

        return self.results


    def update_tokens_from_packet_buffer(self, modality) -> None:
        """
        Tokenize samples from packets of a modality, pads incomplete tokens,
        and updates token availability and completion flags.

        Args:
            modality (str): Modality name, e.g., 'audio' or 'video'.

        Updates:
            Updates `available`, `completed`, and `samples` fields in the token buffer.

        Returns:
            None
        """
        token_duration = self.config['global']['token_duration']
        sampling_rate = self.config['modalities'][modality]['sampling_rate']
        packet_duration = self.config['modalities'][modality]['packet_duration']

        packet_buffer = self.state[modality]['packet_buffer']

        expected_pkt_count = int(token_duration / packet_duration)

        # Identify token indices that are not yet complete
        incomplete_tokens = [
            idx for idx, token in self.state[modality]['token_buffer'].items() if not token['completed']
        ]

        if not incomplete_tokens:
            return  # All tokens completed, nothing to update

        # Filter token intervals to only those incomplete tokens
        incomplete_intervals = [
            self.state['token_intervals'][idx] for idx in incomplete_tokens
        ]

        # Map packets to only incomplete tokens by PTS time
        token_samples_map = {idx: [] for idx in incomplete_tokens}

        for pkt in packet_buffer:
            pkt_time = pkt.pts_time
            for token_idx, (start, end) in zip(incomplete_tokens, incomplete_intervals):
                if start <= pkt_time < end:
                    token_samples_map[token_idx].append(pkt.payload)
                    break

        # Update only incomplete tokens
        for token_idx in incomplete_tokens:
            collected_samples = token_samples_map[token_idx]
            token_entry = self.state[modality]['token_buffer'][token_idx]

            if not collected_samples:
                # No new samples collected, do not update samples but mark as available if previously had samples
                token_entry['available'] = token_entry['samples'] is not None
                continue

            # Concatenate samples
            if modality == 'audio':
                samples = np.concatenate(collected_samples, axis=0)
            else:
                samples = np.array(collected_samples)

            # Determine completion status (only complete if exact number of packets present)
            completed = (len(collected_samples) == expected_pkt_count)

            # Zero-pad if incomplete
            if not completed:
                total_samples = int(token_duration * sampling_rate)
                current_samples_count = len(samples) if modality == 'audio' else samples.shape[0]
                pad_width = total_samples - current_samples_count
                if pad_width > 0:
                    if modality == 'audio':
                        samples = np.pad(samples, (0, pad_width), mode='constant', constant_values=0)
                    else:
                        fallback = np.asarray(self.fallbacks[modality])
                        pad_samples = np.tile(fallback, (pad_width,) + (1,) * (fallback.ndim))
                        samples = np.concatenate((samples, pad_samples), axis=0)

            token_entry['samples'] = samples
            token_entry['available'] = True
            token_entry['completed'] = completed


    def streaming_pipeline(self, modality) -> None:
        """
        Process available tokens of a modality through the streaming feature extraction pipeline.

        Args:
            modality (str): Modality name, e.g., 'audio' or 'video'.

        Updates:
            Marks tokens as `done` and updates their embeddings.

        Returns:
            None
        """
        pipeline = self.pipelines[modality]
        num_tokens = self.config['global']['num_tokens']
        sampling_rate = self.config['modalities'][modality]['sampling_rate']
        device = self.device

        # Select tokens to process: those available but not yet done (batch processing)
        tokens_to_process = [
            (idx, self.state[modality]['token_buffer'][idx]['samples'])
            for idx in range(num_tokens)
            if self.state[modality]['token_buffer'][idx]['available'] and not self.state[modality]['token_buffer'][idx]['done']
        ]

        if modality == 'audio':
            if not tokens_to_process:
                return

            # Convert all raw samples to spectrograms in a list
            spectrograms = [
                waveform_to_examples(data=samples, sample_rate=sampling_rate, return_tensor=False)
                for _, samples in tokens_to_process
            ]

            # Stack and convert to torch tensor batch [batch_size, ...]
            spectrograms_tensor = torch.tensor(np.stack(spectrograms), device=device, dtype=torch.float32)

            # Extract embeddings in batch
            with torch.no_grad():
                emb_batch = pipeline(spectrograms_tensor, return_embs=True)['embs']  # Assume shape [batch_size, ...]
            emb_batch_np = emb_batch.cpu().numpy().astype(np.float32)
           
            # Assign embeddings and update flags
            for (idx, _), emb in zip(tokens_to_process, emb_batch_np):
                token_entry = self.state[modality]['token_buffer'][idx]
                token_entry['embedding'] = np.squeeze(emb)
            
                if token_entry['completed']:
                    token_entry['done'] = True

        elif modality == 'video':
            if not tokens_to_process:
                return

            # Stack samples: shape [batch_size, frames, ...]
            samples_batch = np.stack([samples for _, samples in tokens_to_process])

            # Take temporal mean per token (e.g. mean over frames axis=1)
            emb_batch = samples_batch.mean(axis=1, keepdims=True)  # shape [batch_size, 1, ...]
            
            for (idx, _), emb in zip(tokens_to_process, emb_batch):
                token_entry = self.state[modality]['token_buffer'][idx]
                token_entry['embedding'] = np.squeeze(emb)
                
                if token_entry['completed']:
                    token_entry['done'] = True


    def clear_packet_buffer(self, modality) -> None:
        """
        Remove packets from the packet buffer belonging to tokens that have been completed.

        Args:
            modality (str): Modality name, e.g., 'audio' or 'video'.

        Updates:
            Update packet buffer removing packets associated with completed tokens.  

        Returns:
            None
        """
        packet_buffer = self.state[modality]['packet_buffer']
        token_buffer = self.state[modality]['token_buffer']

        # Get end_time of the latest completed token (or None if none completed)
        completed_tokens_end_times = [
            token['end_time'] for token in token_buffer.values() if token['completed']
        ]

        if not completed_tokens_end_times:
            # No completed tokens, nothing to clear
            return

        max_completed_end_time = max(completed_tokens_end_times)

        # Remove packets with pts_time less than max_completed_end_time
        # (Packets belonging to completed tokens)
        new_packet_buffer = deque()
        for pkt in packet_buffer:
            if pkt.pts_time >= max_completed_end_time:
                new_packet_buffer.append(pkt)

        self.state[modality]['packet_buffer'] = new_packet_buffer


    def get_embeddings(self) -> dict:
        """
        Collect embeddings for all tokens across all modalities into tensors for inference.

        Returns:
            dict: Mapping from modality names to torch.Tensor of shape [num_tokens, embedding_dim, ...].
        """
        embeddings = {}
        device = self.device
        num_tokens = self.config['global']['num_tokens']

        for modality in self.config['modalities'].keys():
            emb_list = []
            for token_idx in range(num_tokens):
                token_entry = self.state[modality]['token_buffer'][token_idx]
                emb = token_entry.get('embedding', None)
                if emb is None:
                    # Use fallback embedding if none available
                    fallback_emb = self.fallbacks[modality]
                    emb = fallback_emb
                emb_list.append(torch.as_tensor(emb, device=device, dtype=torch.float32))

            # Stack embeddings along first dimension (tokens)
            embeddings[modality] = torch.stack(emb_list, dim=0)

        return embeddings
    

    def compute_num_missed_packets(self):
        """
        Calculate the count of missed packets per modality remaining in received streams after inference.

        Updates:
            self.results['num_missed_pkts'][modality] with integer count of missed packets.
        """
        for modality in self.config['modalities'].keys():

            # Remaining packets in the received_stream buffer are missed ones
            remaining_packets = self.received_streams.get(modality, [])
            num_missed = len(remaining_packets) if remaining_packets is not None else 0
            self.results['num_missed_pkts'][modality] = num_missed