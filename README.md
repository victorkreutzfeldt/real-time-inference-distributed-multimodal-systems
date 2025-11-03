# Real-Time Inference for Distributed Multimodal Systems under Communication Delay Uncertainty

This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the paper mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, J. H. Inacio de Souza, S. R. Pandey, B. Soret, and P. Popovski, “Real-Time Inference for Distributed Multimodal Systems under Communication Delay Uncertainty,” submitted to ICC 2026.

<!-- A pre-print version is available on arXiv: [https://arxiv.org/abs/2304.10858](https://arxiv.org/abs/2404.14236). -->

We hope this content helps in your research and contributes to building the precepts behind open science. Remarkably, to boost the idea of open science and further drive the evolution of science, we also motivate you to share your published results with the public.

If you have any questions or if you have encountered any inconsistencies, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
Connected cyber-physical systems perform inference based on real-time inputs from multiple data streams. Uncertain communication delays across data streams challenge the temporal flow of the inference process. State-of-the-art (SotA) non-blocking inference methods rely on a *reference-modality paradigm*, requiring one modality input to be fully received before processing, while depending on costly offline profiling. We propose a novel, *neuro-inspired non-blocking inference paradigm* that primarily employs adaptive temporal windows of integration (TWIs) to dynamically adjust to stochastic delay patterns across heterogeneous streams while relaxing the reference-modality requirement. Our communication-delay-aware framework achieves robust real-time inference with finer-grained control over the accuracy--latency tradeoff. Experiments on the audio-visual event localization (AVEL) task demonstrate superior adaptability to network dynamics compared to SotA approaches.

## Content


## Citing this Repository and License
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

## Acknowledgement
This work was supported by the Villum Investigator Grant “WATER” from the Velux Foundation, Denmark, and by the SNS JU project 6G-GOALS under the EU's Horizon Europe program under Grant Agreement No. 101139232.
