# Real-Time Inference for Distributed Multimodal Systems under Communication Delay Uncertainty
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- 
[![arXiv](https://img.shields.io/badge/arXiv-2506.23118-b31b1b.svg)](https://arxiv.org/abs/2404.14236)-->

This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the paper mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, J. H. Inacio de Souza, S. R. Pandey, B. Soret, and P. Popovski, **“Real-Time Inference for Distributed Multimodal Systems under Communication Delay Uncertainty,”** submitted to ICC 2026.

<!-- A pre-print version is available on arXiv: [https://arxiv.org/abs/2304.10858](https://arxiv.org/abs/2404.14236). -->

We hope this content helps in your research and contributes to building the precepts behind open science. Remarkably, to boost the idea of open science and further drive the evolution of science, we also motivate you to share your published results with the public.

If you have any questions or if you have encountered any inconsistencies, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## 📖 Abstract
Connected cyber-physical systems perform inference based on real-time inputs from multiple data streams. Uncertain communication delays across data streams challenge the temporal flow of the inference process. State-of-the-art (SotA) non-blocking inference methods rely on a *reference-modality paradigm*, requiring one modality input to be fully received before processing, while depending on costly offline profiling. We propose a novel, *neuro-inspired non-blocking inference paradigm* that primarily employs adaptive temporal windows of integration (TWIs) to dynamically adjust to stochastic delay patterns across heterogeneous streams while relaxing the reference-modality requirement. Our communication-delay-aware framework achieves robust real-time inference with finer-grained control over the accuracy-latency tradeoff. Experiments on the audio-visual event localization (AVEL) task demonstrate superior adaptability to network dynamics compared to SotA approaches.

## 📁 Project Structure

```
real-time-inference-distributed-multimodal-systems/
├── data/                       # Directory where all types of data are stored
├── models/                     # Directory for saving trained models
├── src/                        # Source code directory
├── _apply_class_pipeline_audio.py   # Applies audio pipeline over video observations to extract auditory features and embeddings
├── _apply_class_pipeline_video.py   # Applies video pipeline over video observations to extract visual features and embeddings
├── _train.py                  # Trains the baseline model for the AVEL task using extracted auditory embeddings and visual features
├── packetization.py           # Packetizes each video observation into separate auditory and visual streams
├── plotting_per_snr.py        # Plots the performance curves reported in the paper for each SNR value
└── wrapper.py                 # Simulates the wrapper operation under packet delay conditions
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/victorkreutzfeldt/real-time-inference-distributed-multimodal-systems.git
cd real-time-inference-distributed-multimodal-systems
```

2. Create and activate a pip environment using the requirements.txt file:
```bash
python3 -m venv realtime
source realtime/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install the AVE dataset from https://github.com/YapengTian/AVE-ECCV18:

Please, download the dataset, extract and place it under the `data\` folder, having the video observations saved under `data\AVE`
```bash
https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view
```

## 🎯 Usage

### Handling the AVE Dataset

After ensuring the AVE dataset is correctly downloaded and placed in the appropriate folder, run the following preprocessing scripts:

- `data/preprocess.py`: Processes the videos to standardize the auditory stream to a target sampling rate of 16,000 Hz (mono) and the visual stream to 16 FPS with a resolution of 224×224 pixels. Each video clip is trimmed to last 10 seconds. These pre-processed videos will be stored under a new folder `data/AVE_trimmed` with `.avi` format.

- `data/generate_annotations.py`: Generates the corresponding annotation files required for training and evaluation.

### Training Baseline Model

With the dataset prepared (`data/AVE_trimmed`) and annotations generated, you can train the baseline model for the fully-supervised Audio-Visual Event Localization (AVEL) task, based on the implementation from [Yapeng Tian's AVE-ECCV18 repository](https://github.com/YapengTian/AVE-ECCV18). Follow these steps:

- **`_apply_class_pipeline_audio.py`**: Applies the auditory pipeline to extract audio features and embeddings using the VGGish model. This implementation builds upon the [torchvggish project](https://github.com/harritaylor/torchvggish/).

- **`_apply_class_pipeline_visual.py`**: Applies the visual pipeline to extract features using the VGG-19 model. 

Raw data is stored under the folder `data/raw`, while extracted features and embeddings are saved in `data/classification`.

For audio processing, we use correlated spectrograms over mono audio with the following parameters:

- `HOP_LENGTH` = 10 ms  
- `WINDOW_LENGTH` = 25 ms  

This configuration produces **96 spectrogram frames per 1-second token**. For inference, **128-dimensional embeddings** are generated per token after passing through the VGGish pipeline.

For video processing, each 1-second token consists of **16 frames**. Frames are processed through the **VGG-19** network using pixel-stats extracted by running `data/pixel-stats` over the raw videos, and features are pooled as follows:

- **Temporal pooling** averages features across the 16 frames of the token.  
- **Spatial pooling** averages across the 7×7 spatial dimensions of the VGG-19 output (512×7×7), resulting in compact feature representations.

Finally, having the features and embeddings in place, train a model to tackle the AVEL task by running:

- **`_train.py`**: Trains a specified baseline model on the AVEL task using auditory embeddings and visual features generated by the above pipelines.

To obtain the baseline model weights used in the paper, run:
```bash
https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view
```
The `.pth` file for the baseline model is provided in the repository.

### Getting Packets

To simulate streaming transmission of video observations from two different sources, we apply a packetization pipeline over each video observation in the `data/AVE_trimmed` folder, breaking each observation into an auditory and a visual stream, using a `Packet` class defined in `src/packets.py`. Each packetized video observation is stored under the folder `data/packets` with separate folders for audio and video, obtained by running:
```bash
python packetization.py
```

### Simulating Wrapper Operation

To reproduce the curves reported in the paper, run the `wrapper.py` script. This script loads the baseline model, utilizes the packetized data, and operates the auditory and visual pipelines in streaming mode. The script simulates the neuro-inspired wrapper based on the TWI optimization variants described in the paper. When executing the wrapper, specify:

- **`variant`**: Choose the TWI optimization variant, either **PaMo** or **ToMo**.
- **`snr_dB`**: Define the signal-to-noise ratio (SNR) in decibels, with valid options being **-5**, **1.1888**, and **2**, as reported in the paper.

Example command:
```bash
python wrapper.py --variant PaMo --snr_dB -5
```

Each run produces a `.gz` file containing the curve data that depicts inference performance evolution over time. After collecting multiple `.gz` files for different variants and SNR values, use the `plot_per_snr.py` script to generate the curves reported in the paper. Running, for example:
```bash
python wrapper.py--snr_dB -5
```
will generate the `.txt` files suitable for TikZ plotting.

## 📝 Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider citing our work.

<!--
```bibtex
  @INPROCEEDINGS{10901782,
  author={Thorsager, Mathias and Croisfelt, Victor and Shiraishi, Junya and Popovski, Petar},
  booktitle={GLOBECOM 2024 - 2024 IEEE Global Communications Conference}, 
  title={EcoPull: Sustainable IoT Image Retrieval Empowered by TinyML Models}, 
  year={2024},
  volume={},
  number={},
  pages={5066-5071},
  keywords={Energy consumption;Image coding;Biological system modeling;Tiny machine learning;Image retrieval;Mathematical models;Data models;Numerical models;Internet of Things;Data communication;IoT Networks;TinyML;image retrieval;generative AI;medium access control},
  doi={10.1109/GLOBECOM52923.2024.10901782}
}
```
-->

## 🙏 Acknowledgement
This work was supported by the Villum Investigator Grant “WATER” from the Velux Foundation, Denmark, and by the SNS JU project 6G-GOALS under the EU's Horizon Europe program under Grant Agreement No. 101139232.
