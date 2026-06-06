# SOTA - State-of-the-Art Neural Network Modules

[![English](https://img.shields.io/badge/README-English-blue)](README.md)
[![中文](https://img.shields.io/badge/README-中文-red)](README_CN.md)

A systematically organized open-source repository of commonly used neural network modules, with a focus on computer vision tasks.

## Overview

This project collects and implements popular plug-and-play neural network modules for computer vision, covering image classification, object detection, semantic segmentation, and more. Each module includes clear documentation on its paper background, architectural design, PyTorch implementation, and test code. **All modules are originally designed.**

## Directory Structure

```
SOTA/
├── README.md
├── README_CN.md
├── resnet_insert_example.py
└── blocks/
    ├── SRM/  Selective Response Module
    ├── DFA/  Differential Feature Amplifier
    ├── CIM/  Contextual Information Modulator
    ├── GFF/  Gated Feature Fusion
    ├── DRS/  Dynamic Receptive Field Selector
    ├── AFM/  Adaptive Frequency Modulation
    ├── PFA/  Progressive Feature Aggregator
    ├── SAM/  Spatial Affinity Module
    ├── CRM/  Channel Recalibration Module
    ├── LCR/  Local Context Reconstructor
    ├── RIM/  Recursive Inference Module
    ├── PDR/  Polarized Dual Representation
    ├── SSM/  Saliency-Guided Suppression Module
    ├── PGM/  Progressive Gating Module
    ├── FEM/  Feature Equilibrium Module
    ├── IGM/  Information Gathering Module
    ├── RGM/  Reciprocal Guidance Module
    ├── DSM/  Dual-Scale Modulator
    ├── OEM/  Order-Statistic Enhancement Module
    ├── MPM/  Momentum Propagation Module
    ├── PCM/  Phase-Coherence Module
    ├── CGM/  Conditional Gating Module
    ├── IRM/  Information Routing Module
    ├── BSM/  Bilateral Similarity Module
    ├── DGM/  Diversity-Guided Module
    ├── SUM/  Spatial Uncertainty Module
    ├── AGM/  Adaptive Granularity Module
    ├── RAM/  Residual Amplification Module
    ├── TCM/  Tensor Completion Module
    ├── NLM/  Non-local Modulation Module
    ├── EDM/  Entropy-Driven Module
    ├── FIM/  Frequency Importance Module
    ├── HTM/  Hierarchical Transformation Module
    ├── WAM/  Weighted Attention Module
    ├── RCM/  Recursive Convolution Module
    ├── CAM/  Contrast-Aware Module
    ├── SDM/  Spectral Decomposition Module
    ├── QEM/  Quantile Enhancement Module
    └── ...
```

## Implemented Modules

| Date | Module | Full Name | Core Idea | Applications |
|------|--------|-----------|-----------|-------------|
| 05-27 | SRM | Selective Response Module | Group-wise statistics → position-sensitive channel modulation → soft-threshold sparsification | Classification/Detection/Segmentation |
| 05-27 | DFA | Differential Feature Amplifier | Local neighborhood differences → difference-driven amplification → contrast sensitivity | Classification/Detection/Edge Detection |
| 05-27 | CIM | Contextual Information Modulator | Dual-path (local + context) → spatially-adaptive mixing ratio | Classification/Detection/Segmentation |
| 05-27 | GFF | Gated Feature Fusion | Three parallel transformations → joint spatial-channel gating → competitive fusion | Classification/Detection/Segmentation |
| 05-27 | DRS | Dynamic Receptive Field Selector | Multi-dilation parallel branches → spatially-adaptive receptive field selection | Detection/Segmentation (multi-scale) |
| 05-27 | AFM | Adaptive Frequency Modulation | Multi-kernel parallel → frequency band decomposition → spatially-adaptive frequency modulation | Classification/Detection/Image Restoration |
| 05-27 | PFA | Progressive Feature Aggregator | Two-stage coarse-fine tuning → inter-stage information bridge → residual accumulation | Classification/Detection/Segmentation |
| 05-27 | SAM | Spatial Affinity Module | Low-rank projection → affinity matrix → information propagation → global context | Segmentation/Detection/Generation |
| 05-27 | CRM | Channel Recalibration Module | Activation entropy estimation → entropy-guided channel evaluation → redundancy suppression | Classification/Detection/Segmentation |
| 05-27 | LCR | Local Context Reconstructor | Per-position dynamic neighborhood weights → dedicated local convolution kernel → adaptive aggregation | Classification/Detection/Segmentation |
| 05-28 | RIM | Recursive Inference Module | Weight-shared recursive transformation → iterative embedding → residual accumulation refinement | Classification/Detection/Segmentation |
| 05-28 | PDR | Polarized Dual Representation | Spatial/semantic dual pathways → cross gating → polarized feature fusion | Classification/Detection/Segmentation |
| 05-28 | SSM | Saliency-Guided Suppression Module | Saliency detection → adaptive threshold → soft suppression → information budget reallocation | Classification/Detection/Segmentation |
| 05-29 | PGM | Progressive Gating Module | Three-stage cascaded gating (coarse→medium→fine) → channel-wise adaptive fusion | Classification/Detection/Segmentation |
| 05-29 | FEM | Feature Equilibrium Module | Channel statistics encoding → equilibrium energy learning → exponential smoothing adjustment | Classification/Detection/Segmentation |
| 05-29 | IGM | Information Gathering Module | Multi-scale depthwise separable gathering → spatially-adaptive scale weights | Classification/Detection/Segmentation |
| 05-30 | RGM | Reciprocal Guidance Module | Channel-spatial dual-branch reciprocal guidance → bidirectional information modulation | Classification/Detection/Segmentation |
| 05-30 | DSM | Dual-Scale Modulator | Coarse-fine dual-scale mutual modulation → context guidance + detail reinjection | Classification/Detection/Segmentation |
| 05-30 | OEM | Order-Statistic Enhancement Module | Multi-order statistic extraction → spatially-adaptive statistic selection → robust enhancement | Classification/Detection/Segmentation |
| 05-31 | MPM | Momentum Propagation Module | Large-kernel smoothed momentum reference → transient deviation awareness → adaptive modulation | Classification/Detection/Segmentation |
| 05-31 | PCM | Phase-Coherence Module | FFT frequency domain decoupling → amplitude recalibration + phase coherence enhancement | Classification/Detection/Segmentation |
| 05-31 | CGM | Conditional Gating Module | Learnable conditional prototypes → similarity-driven gating → semantic reference enhancement | Classification/Detection/Segmentation |
| 06-01 | IRM | Information Routing Module | Multi-expert content-aware routing → spatially-adaptive mixture of experts | Classification/Detection/Segmentation |
| 06-01 | BSM | Bilateral Similarity Module | K×K neighborhood content similarity → bilateral adaptive weighted aggregation | Classification/Detection/Segmentation |
| 06-01 | DGM | Diversity-Guided Module | Channel Gram matrix → redundancy score-driven modulation → diversity enhancement | Classification/Detection/Segmentation |
| 06-03 | SUM | Spatial Uncertainty Module | Local uncertainty estimation → uncertainty-guided smoothing/preservation dual-path fusion | Classification/Detection/Segmentation |
| 06-03 | AGM | Adaptive Granularity Module | Granularity preference map → coarse-fine dual branch → spatially-adaptive granularity interpolation | Classification/Detection/Segmentation |
| 06-03 | RAM | Residual Amplification Module | Base-residual decomposition → residual information analysis → content-aware amplification/suppression | Classification/Detection/Segmentation |
| 06-06 | TCM | Tensor Completion Module | Low-rank tensor decomposition → signal subspace completion → structure-preserving fusion | Classification/Detection/Segmentation |
| 06-06 | NLM | Non-local Modulation Module | QKV projection → non-local affinity → modulation signal generation → residual modulation | Classification/Detection/Segmentation |
| 06-06 | EDM | Entropy-Driven Module | Local information entropy estimation → entropy-guided enhancement/compression → adaptive resource allocation | Classification/Detection/Segmentation |
| 06-06 | FIM | Frequency Importance Module | DCT frequency domain decomposition → frequency importance learning → spectral recalibration → IDCT | Classification/Detection/Segmentation |
| 06-06 | HTM | Hierarchical Transformation Module | Three-stage progressive transformation → information bridging → adaptive stage fusion | Classification/Detection/Segmentation |
| 06-06 | WAM | Weighted Attention Module | Four-mode parallel attention → mode fusion weight learning → adaptive combination | Classification/Detection/Segmentation |
| 06-06 | RCM | Recursive Convolution Module | Weight-shared recursive convolution → termination gate → spatially-adaptive processing depth | Classification/Detection/Segmentation |
| 06-06 | CAM | Contrast-Aware Module | Local contrast estimation → sharpening/smoothing dual path → contrast-guided fusion | Classification/Detection/Segmentation |
| 06-06 | SDM | Spectral Decomposition Module | Channel covariance spectral decomposition → subspace separation → spectral-domain adaptive filtering | Classification/Detection/Segmentation |
| 06-06 | QEM | Quantile Enhancement Module | Quantile estimation → robust quantile normalization → distribution-aware enhancement | Classification/Detection/Segmentation |

## Usage

Each module can be used as a plug-and-play component embedded into existing networks:

```python
from blocks.RIM.rim import RIM
import torch

rim = RIM(channels=64, num_iterations=3)
x = torch.randn(1, 64, 32, 32)
out = rim(x)
print(out.shape)  # [1, 64, 32, 32]
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- thop (optional, for FLOPs statistics)

## Contributing

Pull requests for new neural network modules are welcome. Please refer to the format of existing modules under `blocks/` and ensure complete documentation and test code are included.

## License

MIT License
