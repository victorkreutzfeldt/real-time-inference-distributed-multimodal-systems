import io
import os

import warnings

import numpy as np
import torch
import h5py

import ffmpeg

from PIL import Image

import matplotlib.pyplot as plt

from src.parameters import *

from src.vggish_input import waveform_to_examples

class AVEDatasetRawFull(object):
    """
    Load AVE dataset with raw features for audio and video.

    Args:
        batch size : int
            Size of the batch to be loaded.
    
    Returns:
        Class to Load AVE dataset. There is a need for global variables, as
        they give the paths to important directories.
    """
    def __init__(self, batch_size, **kwargs):
        super(AVEDatasetRawFull, self).__init__()

        # Get paths 
        self.data_path = kwargs.get('DATA_PATH', DATA_PATH)
        self.labels_path = kwargs.get('LABELS_PATH', LABELS_PATH)
        self.annotations_path =  kwargs.get('ANNOTATIONS_PATH', ANNOTATIONS_PATH)

        # Store batch size
        self.batch_size = batch_size

        # Get labels and extract number of classes
        with h5py.File(self.labels_path, 'r') as hf:
            self.labels = hf['avadataset'][:]
        self.num_classes = int(self.labels.shape[-1])

        # Get annotations
        with open(self.annotations_path, 'r') as file:
            annotations = file.readlines()
        self.annotations = annotations

        # Get size of the dataset 
        self.size = len(self.annotations)

        # Dataset parameters
        self.duration = kwargs.get('DURATION', DURATION)

        # Audio parameters
        self.audio_sample_rate = kwargs.get('AUDIO_SAMPLE_RATE', AUDIO_SAMPLE_RATE)
        self.audio_max_num_patches_total = kwargs.get('AUDIO_MAX_NUM_PATCHES_TOTAL', AUDIO_MAX_NUM_PATCHES_TOTAL)
        self.audio_max_num_samples_total = kwargs.get('AUDIO_MAX_NUM_SAMPLES_TOTAL', AUDIO_MAX_NUM_SAMPLES_TOTAL)
        self.audio_required_num_samples_total = kwargs.get('AUDIO_REQUIRED_NUM_SAMPLES_TOTAL', AUDIO_REQUIRED_NUM_SAMPLES_TOTAL)
        self.audio_num_patches = kwargs.get('AUDIO_NUM_PATCHES', AUDIO_NUM_PATCHES)
        self.audio_num_channels = kwargs.get('AUDIO_NUM_CHANNELS', AUDIO_NUM_CHANNELS)
        self.audio_max_num_patches_model = kwargs.get('AUDIO_MAX_NUM_PATCHES_MODEL', AUDIO_MAX_NUM_PATCHES_MODEL)  
        self.audio_num_bands = kwargs.get('AUDIO_NUM_BANDS', AUDIO_NUM_BANDS) 

        # Video parameters
        self.video_sample_rate = kwargs.get('VIDEO_SAMPLE_RATE', VIDEO_SAMPLE_RATE)
        self.video_max_num_images_total = kwargs.get('VIDEO_MAX_NUM_IMAGES_TOTAL', VIDEO_MAX_NUM_IMAGES_TOTAL)
        self.video_width = kwargs.get('VIDEO_WIDTH', VIDEO_WIDTH)
        self.video_height = kwargs.get('VIDEO_HEIGHT', VIDEO_HEIGHT)
        self.video_num_channels = kwargs.get('VIDEO_NUM_CHANNELS', VIDEO_NUM_CHANNELS)
       #self.video_pix_fmt = kwargs.get('VIDEO_PIX_FMT', VIDEO_PIX_FMT)

    def get_audio_stream(self, file_path):
        """ 
        Get audio stream using the FFMPEG suite. The audio is resampled according to PCM.

        The stdout is set to wait for a sequence of images as 'mjepg',
        that is, a sequence of RGB-formatted JPEG images.
        
        """

        # Probing stream
        probe = ffmpeg.probe(file_path, select_streams='a', show_entries='stream=duration', v='error')
        start_time = float(probe['format']['start_time'])
        
        # Read audio with raw PCM 
        out, _ = (
            ffmpeg
            .input(file_path)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=self.audio_num_channels, ar=self.audio_sample_rate)
            .run(capture_stdout=True, capture_stderr=True)
        )

        # Extract stream
        audio_stream = np.frombuffer(out, dtype=np.int16)

        # Compute pad width and check if it is need to pad
        pad_width = self.audio_max_num_samples_total - len(audio_stream)
        if pad_width > 0:
            if start_time > 0.0:
                audio_stream = np.pad(audio_stream, (pad_width, 0), mode='constant')
            else:
                audio_stream = np.pad(audio_stream, (0, pad_width), mode='constant')
        
        # Clip features
        audio_stream = audio_stream[:self.audio_max_num_samples_total]

        # Pad zeros to ensure that we get the right amount of complete frames
        audio_stream = np.pad(audio_stream, pad_width=(self.audio_required_num_samples_total - self.audio_max_num_samples_total) // 2,  mode='reflect')
        
        # Normalize to [-1.0, 1.0]
        audio_stream = audio_stream.astype(np.float32)
        audio_stream /= np.iinfo(np.int16).max 

        # Extract log-spectogram features
        audio_stream = waveform_to_examples(audio_stream, self.audio_sample_rate)

        # Clip to fit common VGG-ish model input: take half from the start/end
        take_out_patches = self.audio_num_patches - self.audio_max_num_patches_model
        take_out_patches = take_out_patches // 2
        audio_stream = audio_stream[:, :, take_out_patches:, :]
        audio_stream = audio_stream[:, :, :-take_out_patches, :]
        
        return audio_stream
    

    def get_video_stream(self, file_path):
        """
        Get video stream using the FFMPEG suite. The video is resampled. 
        The stdout is set to wait for a sequence of images as 'mjepg',
        that is, a sequence of RGB-formatted JPEG images.

        The duration of a sequence of images is fixed and equal to  DURATION. 
        If the sequence has more images than expected the images are discarded. 
        If the sequence has less images than expected the sequence is padded 
        with zero images.

        Args:
            file_path : str
            Path to file to be loaded, expecting '.mp4'.

        Returns:
            video_stream : numpy.array with shape (T,SpS,RGB,HEIGHT,WIDTH)
            Extracted sequence of JPEG images, normalized between [0,1], where 
            T = DURATION [s]; SpS = SAMPLES PER SECOND.
        
        """

        # Read video with new sample rate
        process = (
            ffmpeg
            .input(file_path)
            .filter('fps', fps=self.video_sample_rate)
            .output('pipe:', format='image2pipe', vcodec='mjpeg') 
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        # Read video as a sequence of JPEG images
        frames = []
        while True:
            # Read a full JPEG image
            in_bytes = b''
            while True:
                byte = process.stdout.read(1)
                if not byte:
                    break
                in_bytes += byte
                if in_bytes[-2:] == b'\xff\xd9':  # End of JPEG
                    break

            if not in_bytes:
                break
            
            # Read byte sequence, convert so RGB, and crop to desired HEIGHT and WIDTH
            image = Image.open(io.BytesIO(in_bytes)).convert('RGB').resize((self.video_height, self.video_width))

            # Store image
            frames.append(image)
        process.wait()
        
        # Get video stream as an array
        video_stream = np.array(frames).transpose(0, 3, 1, 2)
       
        # Check if padding is needed and pad if so
        pad_width = self.video_max_num_images_total - video_stream.shape[0] 
        if pad_width > 0:
            padding = np.zeros((pad_width, self.video_num_channels, self.video_height, self.video_width))
            video_stream = np.concatenate((video_stream, padding))
        
        # Get rid of excess
        video_stream = video_stream[:self.video_max_num_images_total]

        # Normalize to [0, 1]
        video_stream = torch.from_numpy(video_stream).float() / np.iinfo(np.uint8).max 

        # Reshape as (T,SpS,RGB,HEIGHT,WIDTH)
        video_stream = video_stream.reshape(
            (self.duration, self.video_sample_rate, self.video_num_channels, self.video_height, self.video_width)
            )
        
        # Return video stream
        return video_stream

    def get_batch(self, batch_index, output_audio=False, output_video=False, output_labels=False):
        """
        Load audio and video streams for a batch of data points. This function MUST be used while looping
          over batches.

        Args:
        batch_index : int
            Keep track of the current batch. 

        output_audio : boolean (False)
            Flag if you would like to output audio.

        output_video : boolean (False)
            Flag if you would like to output video.
        
        out_labels : boolead (False)
            Flag if you would like to output labels.

        Returns:
            audio_stream : torch.tensor with shape (B,T,SpS,AC,NUM_PATCHES,NUM_BINS)

            video_stream : torch.tensor with shape (B,T,SpS,RGB,HEIGHT,WIDTH)

            labels : torch.tensor with shape (B,T,NUM_CLASSES)
        
        B is size of the batch.
        Raise a Warning if output_audio and output_video are both False.
        """

        # Outputs
        if output_audio:
            audio_stream_ = torch.zeros((self.batch_size, self.duration, self.audio_num_channels, self.audio_max_num_patches_model, self.audio_num_bands))

        if output_video:
            video_stream_ = torch.zeros((self.batch_size, self.duration, self.video_sample_rate, self.video_num_channels, self.video_height, self.video_width))
        
        if output_labels:
            labels_ = torch.zeros((self.batch_size, self.duration, self.num_classes))

        # Go over batch
        for ii in range(self.batch_size):

            # Get current index in terms of the whole dataset
            dataset_index = batch_index * self.batch_size + ii
            
            # Get filename and file path
            filename = self.annotations[dataset_index].split("&")[1]
            file_path = os.path.join(self.data_path, filename + '.mp4')

            # Extract stream
            if output_audio:
                audio_stream = self.get_audio_stream(file_path=file_path)

            if output_video:
                video_stream = self.get_video_stream(file_path=file_path)
       
            # Extract label
            if output_labels: 
                labels = self.labels[dataset_index]

            # Store
            if output_audio:
                audio_stream_[ii] = audio_stream

            if output_video:
                video_stream_[ii] = video_stream
            
            if output_labels:
                labels_[ii] = torch.from_numpy(labels).float()

        output = []

        if output_audio:
            output.append(audio_stream_)

        if output_video:
            output.append(video_stream_)

        if output_labels:
            output.append(labels_)

        if len(tuple(output)) == 1:
            output = output[0]

        return output
        
    def __len__(self):
        return self.size





class AVEDataset(object):
    """
    
    """

    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size):

        # Paths
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size

        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:]
        self.lis = order

        with h5py.File(audio_dir, 'r') as hf:
            self.audio_stream = hf['avadataset'][:]

        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
            
        with h5py.File(video_dir, 'r') as hf:
            self.video_stream = hf['avadataset'][:] 

        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.label_batch = np.float32(np.zeros([self.batch_size, 10, 29]))

    def __len__(self):
        return len(self.lis)

    def get_batch(self, idx):

        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_stream[self.lis[id], :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_stream[self.lis[id], :, :]
            self.label_batch[i, :, :] = self.labels[self.lis[id], :, :]

        return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
            self.label_batch).float()

# class AVE_weak_Dataset(object):
#     def __init__(self, video_dir, video_dir_bg, audio_dir , audio_dir_bg, label_dir, label_dir_bg, label_dir_gt, order_dir, batch_size, status):
#         self.video_dir = video_dir
#         self.audio_dir = audio_dir
#         self.video_dir_bg = video_dir_bg
#         self.audio_dir_bg = audio_dir_bg

#         self.status = status
#         # self.lis_video = os.listdir(video_dir)
#         self.batch_size = batch_size
#         with h5py.File(order_dir, 'r') as hf:
#             train_l = hf['order'][:]
#         self.lis = train_l
#         with h5py.File(audio_dir, 'r') as hf:
#             self.audio_stream = hf['avadataset'][:]
#         with h5py.File(label_dir, 'r') as hf:
#             self.labels = hf['avadataset'][:]
#         with h5py.File(video_dir, 'r') as hf:
#             self.video_stream = hf['avadataset'][:]
#         self.audio_stream = self.audio_stream[train_l, :, :]
#         self.video_stream = self.video_stream[train_l, :, :]
#         self.labels = self.labels[train_l, :]

#         if status == "train":
#             with h5py.File(label_dir_bg, 'r') as hf:
#                 self.negative_labels = hf['avadataset'][:]

#             with h5py.File(audio_dir_bg, 'r') as hf:
#                 self.negative_audio_stream = hf['avadataset'][:]
#             with h5py.File(video_dir_bg, 'r') as hf:
#                 self.negative_video_stream = hf['avadataset'][:]

#             size = self.audio_stream.shape[0] + self.negative_audio_stream.shape[0]
#             audio_train_new = np.zeros((size, self.audio_stream.shape[1], self.audio_stream.shape[2]))
#             audio_train_new[0:self.audio_stream.shape[0], :, :] = self.audio_stream
#             audio_train_new[self.audio_stream.shape[0]:size, :, :] = self.negative_audio_stream
#             self.audio_stream = audio_train_new

#             video_train_new = np.zeros((size, 10, 7, 7, 512))
#             video_train_new[0:self.video_stream.shape[0], :, :] = self.video_stream
#             video_train_new[self.video_stream.shape[0]:size, :, :] = self.negative_video_stream
#             self.video_stream = video_train_new

#             y_train_new = np.zeros((size, 29))
#             y_train_new[0:self.labels.shape[0], :] = self.labels
#             y_train_new[self.labels.shape[0]:size, :] = self.negative_labels
#             self.labels = y_train_new
#         else:
#             with h5py.File(label_dir_gt, 'r') as hf:
#                 self.labels = hf['avadataset'][:]
#                 self.labels = self.labels[train_l, :, :]
#         self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
#         self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
#         if status == "train":
#             self.label_batch = np.float32(np.zeros([self.batch_size, 29]))
#         else:
#             self.label_batch = np.float32(np.zeros([self.batch_size,10, 29]))

#     def __len__(self):
#         return len(self.labels)

#     def get_batch(self, idx):
#         for i in range(self.batch_size):
#             id = idx * self.batch_size + i

#             self.video_batch[i, :, :, :, :] = self.video_stream[id, :, :, :, :]
#             self.audio_batch[i, :, :] = self.audio_stream[id, :, :]
#             if self.status == "train":
#                 self.label_batch[i, :] = self.labels[id, :]
#             else:
#                 self.label_batch[i, :, :] = self.labels[id, :, :]
#         return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
#             self.label_batch).float()
