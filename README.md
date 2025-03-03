<p align="center">
    <h1>MDA-RSM: Multi-Directional Adaptive Remote Sensing Mamba for Building Extraction</h1>
</p>

<p align="center">
    <b>English</b> | <a href="README.zh_CN.md">简体中文</a>
</p>


This repository contains the code implementation of the paper **MDA-RSM: Multi-Directional Adaptive Remote Sensing Mamba for Building Extraction**.

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation Guide](#installation-guide)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Environment Setup](#2-environment-setup)
  - [3. Data Download](#3-data-download)
    - [WHU Dataset](#whu-dataset)
    - [Inria Dataset](#inria-dataset)
    - [Massachusetts Dataset](#massachusetts-dataset)
  - [4. Directory Structure](#4-directory-structure)
  - [5. File and Folder Descriptions](#5-file-and-folder-descriptions)
  - [6. License](#6-license)

---

## Introduction

This project implements a multi-directional adaptive remote sensing Mamba (MDA-RSM) for high-precision building extraction on large-scale datasets with linear computational complexity.

![architecture](https://github.com/Joinzm/MDA-RSM/blob/main/resources/architecture.png)

![mda-ss2d](https://github.com/Joinzm/MDA-RSM/blob/main/resources/mda-ss2d.png)

> Main Contributions:
>
> 1) A novel Mamba-based architecture, MDA-RSM, is proposed. MDA-RSM achieves efficient modeling of building features with both global contextual information and local details while preserving the intrinsic characteristics of large-scale VHR remote sensing images under linear computational complexity.
>
> 2) The idea of modeling the importance of Mamba scanning directions is introduced to remote sensing tasks for the first time. Considering the inherent spatial distribution patterns in remote sensing imagery, a Multi-Directional Adaptive Block (MDABlock) is designed, integrating diverse scanning modes and adaptive directional modeling to help the model focus on and retain critical directional features.
> 3) Extensive experiments validate the effectiveness and superiority of MDA-RSM. The results show that modeling the importance of scan directions significantly improves the model’s representation capability, achieving outstanding performance in building extraction.

---

If you find this project helpful, please give us a star ⭐️.

## Installation Guide

The project environment dependencies are specified in `requirements.txt`. You can refer to the configuration process of [VMamba](https://github.com/MzeroMiko/VMamba) for setting up the environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Joinzm/MDA-RSM.git
cd MDA-RSM
```

### 2. Environment Setup

**Step 1**: Follow the environment setup instructions in the [Vmamba project](https://github.com/MzeroMiko/VMamba) to install the "mamba-env" environment. In our implementation, Python version 3.8 is used.

**Step 2**: Run the following command to install the required dependencies.

If you only need to use the model code, this step is not necessary.

```
bash
pip install -r requirements.txt
```

### 3. Data Download

#### WHU Dataset

- Dataset download link: [WHU Dataset](http://gpcv.whu.edu.cn/data/building_dataset.html).

#### Inria Dataset

- Dataset download link: [Inria Dataset](https://project.inria.fr/aerialimagelabeling/).

#### Massachusetts Dataset

- Dataset download link: [Massachusetts Dataset](https://tianchi.aliyun.com/dataset/93425).

### 4. Directory Structure

You need to organize the dataset in the following format:

```
# {DATASET_ROOT} Dataset root directory
├── train
    ├── image
        └── 0001.png
        └── 0002.png
        └── ...
    ├── label
        └── 0001.png
        └── 0002.png
        └── ...
├── val
    ├── image
        └── 0001.png
        └── 0002.png
        └── ...
    ├── label
        ├── 0001.png
        └── 0002.png
        └── ...
├── test
    ├── image
        └── 0001.png
        └── 0002.png
        └── ...
    ├── label
        └── 0001.png
        └── 0002.png
        └── ...
```

### 5. File and Folder Descriptions

The `config` folder contains models and training hyperparameters for different datasets, which can be imported for the training and testing process.

The `model` folder contains the source code for our model, while the `utils` folder contains various utility scripts, including dataset processing and model parameter statistics.

`train.py` is the training script; `inference.py` is the inference script; `requirements.txt` includes all the required dependencies for the project.

### 6. License

This project is licensed under the [Apache 2.0 License](https://apache.ac.cn/licenses/LICENSE-2.0).If you use code or performance benchmarks from this project in your research, please cite MDA-RSM.
