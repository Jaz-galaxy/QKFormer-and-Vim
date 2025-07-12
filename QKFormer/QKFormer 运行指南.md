# QKFormer SNN模型训练指南

## 项目概述

QKFormer是一个突破性的脉冲神经网络(SNN)项目，在ImageNet-1K上首次达到85.65%的准确率，超越了直接训练SNN的历史记录。本指南详细说明了如何在CIFAR-10、CIFAR-100和ImageNet数据集上训练和测试QKFormer模型。

## 环境配置

### 1. 创建虚拟环境

```bash
conda create -n qkformer python=3.8 -y
source activate qkformer
```

### 2. 安装依赖包

```bash
# 安装核心依赖
pip install torch==1.12.1 torchvision==0.13.1
pip install timm==0.6.12
pip install spikingjelly
pip install tensorboard
pip install torchinfo einops

# 注意：如果cupy不兼容，模型会自动使用torch后端
```

### 3. 验证环境

```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## 数据集准备

### CIFAR-10/100 数据集
数据集会在首次运行时自动下载到`./data/`目录下。

### ImageNet 数据集
需要手动准备ImageNet数据，并按以下结构组织：
```
imagenet/data/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ...
```

如果只有训练数据，可以使用提供的分割脚本：
```bash
cd imagenet
python split_dataset.py --data_dir ./data --val_ratio 0.2 --seed 42
```

### Tiny ImageNet 数据集
用于快速测试和原型验证的小规模ImageNet数据集：
```
imagenet/mini_data/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    └── ...
```

可以使用以下命令准备Tiny ImageNet数据：
```bash
cd imagenet
# 如果你有完整的ImageNet数据，可以创建一个子集
python split_dataset.py --data_dir ./data --output_dir ./mini_data --subset_size 100000 --val_size 10000
```

## 训练命令

### 1. CIFAR-10 训练

```bash
cd cifar10
source activate qkformer

# 基础训练命令
python train.py \
    --epochs 400 \
    --batch-size 64 \
    --model QKFormer \
    --time-step 4 \
    --layer 4 \
    --dim 384 \
    --num_heads 8 \
    --patch-size 4 \
    --mlp-ratio 4 \
    --workers 8

# 内存受限时的配置
python train.py \
    --epochs 400 \
    --batch-size 16 \
    --model QKFormer \
    --time-step 4 \
    --layer 4 \
    --dim 192 \
    --num_heads 4 \
    --patch-size 4 \
    --mlp-ratio 4 \
    --workers 2
```

### 2. CIFAR-100 训练

```bash
cd cifar100
source activate qkformer

python train.py \
    --epochs 400 \
    --batch-size 64 \
    --model QKFormer \
    --time-step 4 \
    --layer 4 \
    --dim 384 \
    --num_heads 8 \
    --patch-size 4 \
    --mlp-ratio 4 \
    --num-classes 100 \
    --workers 8
```

### 3. ImageNet 训练

```bash
cd imagenet
source activate qkformer

# 单GPU训练
python train.py \
    --epochs 200 \
    --batch_size 64 \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 224 \
    --data_path ./data

# 多GPU分布式训练
python -m torch.distributed.launch --nproc_per_node=8 train.py \
    --epochs 200 \
    --batch_size 32 \
    --model QKFormer_10_768 \
    --time_step 4 \
    --input_size 224 \
    --data_path ./data
```

### 4. Tiny ImageNet 训练（轻量级测试）

如果你想在小规模数据集上快速测试模型，可以使用Tiny ImageNet训练脚本：

```bash
cd imagenet
source activate qkformer

# 基础训练命令
python train_tiny_imagenet_tqdm.py \
    --epochs 100 \
    --batch_size 64 \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 64 \
    --data_path ./mini_data \
    --num_workers 4

# 快速测试（1轮训练）
python train_tiny_imagenet_tqdm.py \
    --epochs 1 \
    --batch_size 64 \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 64 \
    --data_path ./mini_data \
    --num_workers 2 \
    --warmup_epochs 0

# 内存受限配置
python train_tiny_imagenet_tqdm.py \
    --epochs 100 \
    --batch_size 32 \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 64 \
    --data_path ./mini_data \
    --num_workers 2
```

### 5. ImageNet 测试

```bash
cd imagenet
source activate qkformer

python test.py \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 224 \
    --data_path ./data \
    --resume /path/to/checkpoint.pth
```

## 配置文件说明

### CIFAR-10 配置 (cifar10.yml)
```yaml
epochs: 400
time_step: 4
layer: 4
dim: 384
num_heads: 8
patch_size: 4
mlp_ratio: 4
data_dir: ./data/
dataset: torch/cifar10
num_classes: 10
img_size: 32
batch_size: 64
lr: 1e-3
weight_decay: 6e-2
opt: adamw
sched: cosine
```

### CIFAR-100 配置 (cifar100.yml)
```yaml
epochs: 400
time_step: 4
dim: 384
num_heads: 8
patch_size: 4
mlp_ratio: 4
data_dir: ./data/
dataset: torch/cifar100
num_classes: 100
img_size: 32
batch_size: 64
lr: 1e-3
weight_decay: 6e-2
opt: adamw
sched: cosine
```

## 超参数说明

### 关键参数
- **time_step**: 脉冲时间步数，通常设为4
- **dim**: 嵌入维度，影响模型大小和性能
- **num_heads**: 注意力头数，需要能被dim整除
- **patch_size**: 图像补丁大小
- **mlp_ratio**: MLP扩展比例

### 训练参数
- **epochs**: 训练轮数
- **batch_size**: 批次大小，根据GPU内存调整
- **lr**: 学习率，推荐1e-3
- **weight_decay**: 权重衰减，推荐6e-2
- **opt**: 优化器，推荐adamw
- **sched**: 学习率调度，推荐cosine

## 模型变体

### ImageNet模型变体
- **QKFormer_10_384**: 16.47M参数，78.80%准确率
- **QKFormer_10_512**: 29.08M参数，82.04%准确率
- **QKFormer_10_768**: 64.96M参数，85.65%准确率

### 内存优化建议
- 减小batch_size：从64降到16或更小
- 减小模型维度：从384降到192
- 减少注意力头数：从8降到4
- 使用gradient checkpointing
- 使用混合精度训练(AMP)

## 故障排除

### 1. CUDA内存不足
```bash
# 减小batch size和模型尺寸
python train.py --batch-size 8 --dim 192 --num_heads 4
```

### 2. CuPy后端问题
模型会自动回退到torch后端，性能略有降低但功能正常。

### 3. 数据加载错误
```bash
# 检查数据路径
ls ./data/
# 重新下载数据集
rm -rf ./data/ && python train.py
```

### 4. 训练速度慢
- 增加workers数量：`--workers 8`
- 使用SSD存储数据
- 启用混合精度：`--amp`

## 性能基准

### 预期结果
| 数据集 | 模型配置 | Top-1准确率 | 训练时间 |
|--------|----------|-------------|----------|
| CIFAR-10 | QKFormer-384 | ~95% | 8小时 |
| CIFAR-100 | QKFormer-384 | ~75% | 8小时 |
| Tiny ImageNet | QKFormer-384 | ~25% | 2小时 |
| ImageNet | QKFormer-384 | ~79% | 3天 |

### 硬件要求
- **最低配置**: RTX 3080 (10GB VRAM)
- **推荐配置**: RTX 4090 (24GB VRAM) 
- **内存**: 32GB RAM
- **存储**: 500GB SSD

## 预训练模型

可从以下链接下载预训练模型：
- [QKFormer-16M](https://pan.baidu.com/s/1mX0jQyKZ5p6ZDzvMVeY20A) (密码: abcd)
- [QKFormer-29M](https://pan.baidu.com/s/1luWM1L8gV3BI7REh4MgbkA) (密码: abcd)
- [QKFormer-65M](https://pan.baidu.com/s/1WJW1wC0Vs-lvGjYr5pGV_w) (密码: abcd)
- [Google Drive](https://drive.google.com/drive/folders/1vhq9jmhmuyZ5_RGHuWD4wniza856qF8U)

