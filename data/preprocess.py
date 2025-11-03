import os
import subprocess
from tqdm import tqdm

# ====== Configurations ======
TARGET_FPS = 16
TARGET_SAMPLING_RATE = 16000
TARGET_DURATION = 10.0
TARGET_RESOLUTION = (224, 224)

INPUT_PATH = 'data/AVE'
OUTPUT_PATH = "data/AVE_trimmed"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ====== Utility Functions ======

def clip_crop_resample_video(input_file, output_file, target_duration=10.0, target_fps=16, target_sampling_rate=16000, resolution=(224, 224)):
    """
    Trim/pad video and audio to fixed duration and specs, rescale video, downsample and mono audio.
    Uses direct FFmpeg command line via subprocess.
    """
    width, height = resolution

    # Define audio filter for resampling and mono conversion
    audio_filter = f"aresample=resampler=soxr:cutoff=0.95:precision=28:osf=s16:osr={target_sampling_rate}:dither_method=shibata"

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", "0",
        "-t", str(target_duration),
        "-i", input_path,
        "-vf", f"fps={target_fps},scale={width}:{height}",
        "-af", audio_filter,
        "-ac", "1", 
        "-vcodec", "mjpeg",
        "-acodec", "pcm_s16le",
        "-f", "avi",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print(f"Processed {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to process {input_path}:\n{e.stderr.decode()}")

# ====== Main ======
if __name__ == "__main__":

    # Get video files from input path
    video_files = [f for f in os.listdir(INPUT_PATH) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Iterate and process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_path = os.path.join(INPUT_PATH, video_file)
        video_id = os.path.splitext(video_file)[0]
        output_path = os.path.join(OUTPUT_PATH, video_id + '.avi')

        clip_crop_resample_video(
            input_path, output_path,
            target_duration=TARGET_DURATION,
            target_fps=TARGET_FPS,
            target_sampling_rate=TARGET_SAMPLING_RATE,
            resolution=TARGET_RESOLUTION
        )
