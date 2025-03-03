<p align="center">
    <h1>MDA-RSM: Multi-Directional Adaptive Remote Sensing Mamba for Building Extraction</h1>
</p>


<p align="center">
    <a href="README.md">English</a> | <b>简体中文</b>
</p>

本项目仓库是论文 **MDA-RSM: Multi-Directional Adaptive Remote Sensing Mamba for Building Extraction**的代码实现

---

## 目录

- [目录](#目录)
- [简介](#简介)
- [安装指南](#安装指南)
  - [1.克隆仓库](#1克隆仓库)
  - [2.环境安装](#2环境安装)
  - [3.数据下载](#3数据下载)
    - [WHU 数据集](#whu-数据集)
    - [Inria 数据集](#inria-数据集)
    - [Massachusetts 数据集](#massachusetts-数据集)
  - [4.组织方式](#4组织方式)
  - [5.文件夹与文件说明](#5文件夹与文件说明)
  - [6.许可协议](#6许可协议)

---

## 简介

本项目实现了一种多方向自适应的遥感Mamba（MDA-RSM） ，用于在大规模数据集上进行线性复杂度的高精度建筑物提取。

![architecture](resources\architecture.png)

![mda-ss2d](resources\mda-ss2d.png)

> 主要贡献：
>
> 1)提出了一种新的基于mamba的架构MDA-RSM。MDA-RSM实现了具有全局上下文信息和局部细节的高效建筑物特征建模，同时在线性计算复杂度下保留了大尺度遥感图像的内在特征
>
> 2)首次将Mamba扫描方向重要性建模的思想引入遥感任务中。考虑到遥感影像固有的空间分布模式，设计了一种多方向自适应块MDABlock，融合多种扫描模式和自适应方向建模，帮助模型聚焦和保留关键方向特征
>
> 3)通过大量实验验证了MDA-RSM的有效性和优越性。实验结果表明，对扫描方向的重要性进行建模可以显著提高模型的表示能力，MDA-RSM在建筑物提取方面取得了优异性能。

---

如果你觉得本项目对你有帮助，请给我们一个 star ⭐️。

## 安装指南

项目环境依赖已在Requirement.txt中说明，环境配置可参考 [VMamba](https://github.com/MzeroMiko/VMamba) 的配置过程。

### 1.克隆仓库

```bash
git clone https://github.com/Joinzm/MDA-RSM.git
cd MDA-RSM
```

### 2.环境安装

**步骤 1**：按照[Vmamba项目](https://github.com/MzeroMiko/VMamba)的环境安装指示，安装好"mamba-env"环境。在我们的实现中，python版本为3.8。

**步骤 2**：运行以下命令安装依赖包

如果你只需要使用模型代码，则不需要这一步.

```
pip install -r requirements.txt
```

### 3.数据下载

#### WHU 数据集

- 数据集下载地址： [WHU 数据集](http://gpcv.whu.edu.cn/data/building_dataset.html)。

#### Inria 数据集

- 数据集下载地址： [Inria 数据集](https://project.inria.fr/aerialimagelabeling/)。

#### Massachusetts 数据集

- 数据集下载地址： [Massachusetts 数据集](https://tianchi.aliyun.com/dataset/93425)。

### 4.组织方式

你需要将数据集组织成如下的格式：

```
${DATASET_ROOT} # 数据集根目录，
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
        └──0001.png
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

### 5.文件夹与文件说明

`config`文件夹存放着不同数据集的模型和训练的超参数，它们可以通过import加载到训练和测试过程

`model`文件夹存放着我们的模型源码，`utils`文件夹存放着各类其他的代码文件，包含数据集的处理、模型参数的统计等。

`train.py`是我们的训练代码；`inference.py`是我们的推理代码；`requirements.txt`里面包含了我们项目所需的依赖环境

### 6.许可协议

该项目采用 [Apache 2.0 开源许可证](https://apache.ac.cn/licenses/LICENSE-2.0)。如果你在研究中使用了本项目的代码或者性能基准，请引用MDA-RSM。


