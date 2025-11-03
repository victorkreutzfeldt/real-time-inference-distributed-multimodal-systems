# Real-Time Inference for Distributed Multimodal Systems under Communication Delay Uncertainty
<!-- 
[![arXiv](https://img.shields.io/badge/arXiv-2506.23118-b31b1b.svg)](https://arxiv.org/abs/2404.14236)-->
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the paper mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, J. H. Inacio de Souza, S. R. Pandey, B. Soret, and P. Popovski, **“Real-Time Inference for Distributed Multimodal Systems under Communication Delay Uncertainty,”** submitted to ICC 2026.

<!-- A pre-print version is available on arXiv: [https://arxiv.org/abs/2304.10858](https://arxiv.org/abs/2404.14236). -->

We hope this content helps in your research and contributes to building the precepts behind open science. Remarkably, to boost the idea of open science and further drive the evolution of science, we also motivate you to share your published results with the public.

If you have any questions or if you have encountered any inconsistencies, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## 📖 Abstract
Connected cyber-physical systems perform inference based on real-time inputs from multiple data streams. Uncertain communication delays across data streams challenge the temporal flow of the inference process. State-of-the-art (SotA) non-blocking inference methods rely on a *reference-modality paradigm*, requiring one modality input to be fully received before processing, while depending on costly offline profiling. We propose a novel, *neuro-inspired non-blocking inference paradigm* that primarily employs adaptive temporal windows of integration (TWIs) to dynamically adjust to stochastic delay patterns across heterogeneous streams while relaxing the reference-modality requirement. Our communication-delay-aware framework achieves robust real-time inference with finer-grained control over the accuracy-latency tradeoff. Experiments on the audio-visual event localization (AVEL) task demonstrate superior adaptability to network dynamics compared to SotA approaches.

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/victorkreutzfeldt/real-time-inference-distributed-multimodal-systems.git
cd real-time-inference-distributed-multimodal-systems
```

2. Create and activate the Conda environment using the environment.yml file:
```bash
conda env create -f environment.yml
conda activate realtime
```

3. Install the AVE dataset from https://github.com/YapengTian/AVE-ECCV18:

Please, download the dataset, extract and place it under the `data\` folder.
```bash
https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view
```

## 🎯 Usage

### Handling the AVE Dataset

After ensuring the AVE dataset is correctly downloaded and placed in the appropriate folder, run the following preprocessing scripts:

- `data/preprocess.py`: Processes the videos to standardize the auditory stream to a target sampling rate of 16,000 Hz (mono) and the visual stream to 16 FPS with a resolution of 224×224 pixels. Each video clip is trimmed to last 10 seconds. These pre-processed videos will be stored under a new folder `data/AVE_trimmed` with `.avi` format.

- `data/generate_annotations.py`: Generates the corresponding annotation files required for training and evaluation.

- `data/pixel_stats.py`: Computes pixel-wise statistics across the dataset for normalization purposes during model training.

### Training Baseline Model

With the dataset prepared (`data/AVE_trimmed`) and annotations generated, you can train the baseline model for the fully-supervised Audio-Visual Event Localization (AVEL) task, based on the implementation from [Yapeng Tian's AVE-ECCV18 repository](https://github.com/YapengTian/AVE-ECCV18). Follow these steps:

- **`_apply_class_pipeline_audio.py`**: Applies the auditory pipeline to extract audio features and embeddings using the VGGish model. This implementation builds upon the [torchvggish project](https://github.com/harritaylor/torchvggish/).

- **`_apply_class_pipeline_visual.py`**: Applies the visual pipeline to extract features using the VGG-19 model.

Raw data is stored under the folder `data/raw`, while extracted features and embeddings are saved in `data/classification`.

For audio processing, we use correlated spectrograms over mono audio with the following parameters:

- `HOP_LENGTH` = 10 ms  
- `WINDOW_LENGTH` = 25 ms  

This configuration produces **96 spectrogram frames per 1-second token**. For inference, **128-dimensional embeddings** are generated per token after passing through the VGGish pipeline.

For video processing, each 1-second token consists of **16 frames**. Frames are processed through the **VGG-19** network, and features are pooled as follows:

- **Temporal pooling** averages features across the 16 frames of the token.  
- **Spatial pooling** averages across the 7×7 spatial dimensions of the VGG-19 output (512×7×7), resulting in compact feature representations.

Finally, having the features and embeddings in place, train a model to tackle the AVEL task by running:
- **`_train.py`**: Trains a specified baseline model on the AVEL task using auditory embeddings and visual features generated by the above pipelines.

To obtain the baseline model weights as used in the paper, run:
```bash
https://drive.google.com/file/d/1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK/view
```
The `.pth` for the baseline model is provided in the repo.

### Getting Packets
To simulate transmission of video observations from two different sources, we apply a packetization pipeline over each video observation in the `data/AVE_trimmed` folder, breaking each observation into an auditory and a visual stream, using a `Packet` class defined in `src/packets.py`. Each packetized video observation is store under the folder `data/packets`, obtained by running: 
```bash
packetization.py
```

### Simulating Wrapper Operation




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
