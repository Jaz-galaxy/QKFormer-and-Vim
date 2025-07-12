# Vision Mamba 运行指南

## 项目概述

Vision Mamba是一个基于双向状态空间模型(SSMs)的计算机视觉骨干网络，用双向Mamba块替代了传统的自注意力机制。该项目在ImageNet分类、COCO目标检测和ADE20k语义分割任务上都取得了优异性能，同时具有显著的计算效率优势。

## 环境配置

### 系统要求

- CUDA 12.1+
- Python 3.10.13
- PyTorch 2.1.1
- GPU: 推荐RTX 4090或更高

### 环境安装

```bash
# 1. 创建conda环境
conda create -n vim_env python=3.10.13
conda activate vim_env

# 2. 安装PyTorch (CUDA 12.1版本)
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. 安装项目依赖
pip install -r vim/vim_requirements.txt

# 4. 编译安装核心组件
pip install -e causal-conv1d
pip install -e mamba-1p1p1

# 5. 验证安装
python -c "
import torch
from mamba_ssm.modules.mamba_simple import Mamba
print('✓ 环境配置成功')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
"
```

## 数据准备

### ImageNet-1K数据集结构

```
imagenet_data/
├── train/
│   ├── n01440764/  # 类别1
│   ├── n01443537/  # 类别2
│   └── ...         # 1000个类别
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### 数据验证

```bash
# 检查数据集
ls -la imagenet_data/train/ | wc -l  # 应该显示1001 (1000类别+.和..)
ls -la imagenet_data/val/ | wc -l    # 应该显示1001
```

## 训练命令

### 基础训练命令

```bash
# 进入训练目录
cd vim/

# Vision Mamba Tiny训练
python main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --epochs 300 \
    --lr 5e-4 \
    --data-path /path/to/imagenet_data \
    --output_dir ./output/vim_tiny \
    --num_workers 8 \
    --no_amp
```

### 内存优化版本（推荐）

```bash
# 适合单GPU RTX 4090的配置
python main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 64 \
    --epochs 100 \
    --lr 1e-4 \
    --weight-decay 0.05 \
    --warmup-epochs 5 \
    --data-path /root/autodl-tmp/Vim/imagenet_data \
    --output_dir ./output/vim_tiny_optimized \
    --num_workers 4 \
    --no_amp \
    --drop-path 0.1
```

### 分布式训练（多GPU）

```bash
# 8GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 128 \
    --epochs 300 \
    --lr 5e-4 \
    --data-path /root/autodl-tmp/Vim/imagenet_data \
    --output_dir ./output/vim_tiny_distributed \
    --no_amp
```

## 模型评估

### 评估预训练模型

```bash
python main.py \
    --eval \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --resume /path/to/checkpoint.pth \
    --data-path /root/autodl-tmp/Vim/imagenet_data \
    --batch-size 256
```

### 评估不同输入尺寸

```bash
# 384x384输入尺寸评估
python main.py \
    --eval \
    --model vim_tiny_patch16_384_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --input-size 384 \
    --resume /path/to/checkpoint.pth \
    --data-path /root/autodl-tmp/Vim/imagenet_data
```

## 命令行参数详解

### 核心训练参数

| 参数              | 默认值 | 说明         |
| ----------------- | ------ | ------------ |
| `--model`         | -      | 模型架构名称 |
| `--batch-size`    | 64     | 批次大小     |
| `--epochs`        | 300    | 训练轮数     |
| `--lr`            | 5e-4   | 学习率       |
| `--weight-decay`  | 0.05   | 权重衰减     |
| `--drop-path`     | 0.1    | DropPath比率 |
| `--warmup-epochs` | 5      | 预热轮数     |

### 优化器参数

| 参数          | 默认值 | 说明          |
| ------------- | ------ | ------------- |
| `--opt`       | adamw  | 优化器类型    |
| `--opt-eps`   | 1e-8   | 优化器epsilon |
| `--momentum`  | 0.9    | 动量参数      |
| `--sched`     | cosine | 学习率调度器  |
| `--min-lr`    | 1e-5   | 最小学习率    |
| `--warmup-lr` | 1e-6   | 预热学习率    |

### 数据增强参数

| 参数             | 默认值               | 说明             |
| ---------------- | -------------------- | ---------------- |
| `--color-jitter` | 0.3                  | 颜色抖动强度     |
| `--aa`           | rand-m9-mstd0.5-inc1 | AutoAugment策略  |
| `--mixup`        | 0.8                  | Mixup强度        |
| `--cutmix`       | 1.0                  | CutMix强度       |
| `--reprob`       | 0.25                 | Random Erase概率 |
| `--smoothing`    | 0.1                  | 标签平滑         |

### 模型EMA参数

| 参数                    | 默认值  | 说明         |
| ----------------------- | ------- | ------------ |
| `--model-ema`           | True    | 启用模型EMA  |
| `--model-ema-decay`     | 0.99996 | EMA衰减率    |
| `--model-ema-force-cpu` | False   | 强制EMA在CPU |

### Vision Mamba特有参数

| 参数                             | 默认值 | 说明             |
| -------------------------------- | ------ | ---------------- |
| `--if_random_cls_token_position` | False  | 随机类标记位置   |
| `--if_random_token_rank`         | False  | 随机token顺序    |
| `--no_amp`                       | -      | 禁用混合精度训练 |

## 模型配置参数

### VisionMamba模型参数

| 参数         | Tiny | Small | Base |
| ------------ | ---- | ----- | ---- |
| `embed_dim`  | 192  | 384   | 768  |
| `depth`      | 24   | 24    | 24   |
| `d_state`    | 16   | 16    | 16   |
| `patch_size` | 16   | 16    | 16   |
| `img_size`   | 224  | 224   | 224  |
| 参数量       | 7M   | 26M   | 98M  |

### 关键配置选项

```python
# 模型创建示例
model = VisionMamba(
    img_size=224,           # 输入图像尺寸
    patch_size=16,          # 补丁大小
    depth=24,               # 层数
    embed_dim=192,          # 嵌入维度
    d_state=16,             # 状态维度
    num_classes=1000,       # 类别数
    drop_path_rate=0.1,     # DropPath率
    rms_norm=True,          # 使用RMSNorm
    fused_add_norm=False,   # 禁用融合优化(解决兼容性)
    if_bimamba=True,        # 启用双向Mamba
    bimamba_type="v2",      # Mamba类型
    use_middle_cls_token=True  # 类标记中间位置
)
```

## 常见问题与解决方案

### 1. CUDA内存不足

**错误信息：** `CUDA out of memory`

**解决方案：**

```bash
# 减小批次大小
--batch-size 32  # 从128降到32

# 启用梯度检查点
--use-checkpoint

# 禁用混合精度
--no_amp
```

### 2. Triton兼容性问题

**错误信息：** `Cannot find backend for cpu`

**解决方案：**

```python
# 在模型配置中设置
fused_add_norm=False  # 已在代码中修复
```

### 3. 数据加载速度慢

**解决方案：**

```bash
# 调整workers数量
--num_workers 4  # 根据CPU核心数调整

# 启用内存锁定
--pin-mem
```

### 4. 训练不稳定

**解决方案：**

```bash
# 降低学习率
--lr 1e-4

# 增加预热期
--warmup-epochs 10

# 调整权重衰减
--weight-decay 0.01
```

##  性能优化建议

### 1. 内存优化

```bash
# 推荐配置 (RTX 4090 24GB)
python main.py \
    --batch-size 64 \
    --num_workers 4 \
    --no_amp \
    --pin-mem
```

### 2. 速度优化

```bash
# 使用编译优化
export TORCH_COMPILE=1

# 启用数据预加载
--pin-mem

# 优化数据并行
--num_workers 8
```

### 3. 精度优化

```bash
# 启用模型EMA
--model-ema

# 使用标签平滑
--smoothing 0.1

# 强数据增强
--mixup 0.8 --cutmix 1.0
```

##  预期结果

### ImageNet-1K分类性能

| 模型      | 参数量 | Top-1 准确率 | Top-5 准确率 | 速度提升 | 内存节省 |
| --------- | ------ | ------------ | ------------ | -------- | -------- |
| Vim-Tiny  | 7M     | 76.1%        | 93.0%        | 2.8x     | 86.8%    |
| Vim-Small | 26M    | 80.5%        | 95.1%        | 2.5x     | 84.2%    |
| Vim-Base  | 98M    | 81.9%        | 95.8%        | 2.1x     | 82.5%    |

### 训练时间估算 (RTX 4090)

- **Vim-Tiny**: 约8-12小时 (100 epochs)
- **Vim-Small**: 约16-24小时 (100 epochs)  
- **Vim-Base**: 约32-48小时 (100 epochs)

##  调试模式

### 快速验证训练流程

```bash
# 小规模测试 (1个epoch, 小批次)
python main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 4 \
    --epochs 1 \
    --lr 1e-4 \
    --data-path /root/autodl-tmp/Vim/imagenet_data \
    --output_dir ./output/debug \
    --num_workers 1 \
    --no_amp \
    --warmup-epochs 0
```

### 模型测试

```bash
# 测试模型前向传播
python -c "
import torch
import models_mamba

# 创建模型
model = models_mamba.vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
model = model.cuda()

# 测试输入
x = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    y = model(x)
    print(f'输入: {x.shape}')
    print(f'输出: {y.shape}')
    print('✓ 模型测试成功')
"
```

