# Data path
DATA_PATH = 'data/AVE'
LABELS_PATH = 'data/labels.h5'
ANNOTATIONS_PATH = 'data/annotations.txt'

# 
DURATION = 10 # Total duration in seconds

# Video parameters
VIDEO_SAMPLE_RATE = 16  # image per second

VIDEO_MAX_NUM_IMAGES_TOTAL = int(DURATION * VIDEO_SAMPLE_RATE)

VIDEO_WIDTH = 224
VIDEO_HEIGHT = 224
VIDEO_NUM_CHANNELS = 3  # RGB
VIDEO_PIX_FMT = 'rgb24'

# Audio parameters
AUDIO_SAMPLE_RATE = 16000
AUDIO_NUM_CHANNELS = 1 # Mono

AUDIO_NUM_PATCHES = 100  # Number of patches in input mel-spectrogram patch
AUDIO_NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch
AUDIO_MAX_NUM_PATCHES_MODEL = 96 # Maximum number of patches of NUM_BANDS according to the input to VGG model used for feature extraction

STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010

AUDIO_MAX_NUM_PATCHES_TOTAL = int(DURATION  // STFT_HOP_LENGTH_SECONDS) + 1
AUDIO_MAX_NUM_SAMPLES_TOTAL = int(DURATION * AUDIO_SAMPLE_RATE)

STFT_WINDOW_LENGTH_SAMPLES = int(round(AUDIO_SAMPLE_RATE * STFT_WINDOW_LENGTH_SECONDS))
STFT_HOP_LENGTH_SAMPLES = int(round(AUDIO_SAMPLE_RATE * STFT_HOP_LENGTH_SECONDS))

AUDIO_REQUIRED_NUM_SAMPLES_TOTAL = (AUDIO_MAX_NUM_PATCHES_TOTAL - 1) * STFT_HOP_LENGTH_SAMPLES + STFT_WINDOW_LENGTH_SAMPLES

NUM_MEL_BINS = AUDIO_NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram
EXAMPLE_WINDOW_SECONDS = 1.00  # Each example contains 100 (old:96) 10ms frames
EXAMPLE_HOP_SECONDS = 1.00#0.96  # with zero overlap