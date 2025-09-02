# 地震学深度学习项目使用指南

## 项目概述

这是一个专业的地震学深度学习项目，专门用于地震波形数据的智能分析和处理。项目集成了多种先进的深度学习模型，支持多种地震学任务，包括地震检测、震相识别、震级估计、方位角估计等。

### 主要特性

- 🌊 **多种地震学任务支持**：地震检测、震相识别、震级估计、方位角估计等
- 🤖 **先进的模型架构**：SeisT、EQTransformer、PhaseNet、MagNet、基于LLaMA的模型等
- 📊 **多数据集支持**：DiTing、STEAD、INSTANCE、PNW、CSNCD等
- 🔧 **完整训练流程**：预训练、微调、分布式训练、数据增强
- 📈 **全面评估体系**：多种评估指标、可视化工具、结果分析

## 环境配置

### 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+（GPU训练）
- HDF5库支持

### 安装依赖

```bash
# 创建虚拟环境
conda create -n seismic-dl python=3.8
conda activate seismic-dl

# 安装PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

# 安装其他依赖
pip install h5py pandas numpy matplotlib seaborn
pip install tensorboard scikit-learn
pip install transformers  # 如果使用LLaMA模型
```

## 项目结构

```
.
├── DASEventData 处理过后的数据文件
│   ├── data 3km光纤人工标注标签对应的npy数据 100Hz
│   ├── micro_eq
│   ├── phase_picks 3km光纤人工标注标签
│   └── xfj_sac_data
├── README.md
├── course-xiao 微调使用的代码
│   ├── README.md
│   ├── ckpt
│   ├── course_sft_code_enc 主要代码位置
│   └── data
├── data
│   ├── NewZealand 吴子璇师兄的新西兰数据 
│   ├── taiwan 台湾花莲地震的几个事件数据
│   └── xfj xfj3km光纤数据和处理结果
        ├── DetectedFinal.dat
        ├── das_event_reorganize            npy数据
        ├── das_event_reorganize_fig        无标签元数据
        ├── npy_diting_100Hz                100Hz diting输出结果
        ├── npy_diting_300Hz                300Hz diting输出结果
        ├── phasenet_das_results            phasenet das 结果，马师兄跑的
        ├── phasenet_results                phasenet 结果 罗老师跑的
        ├── phasenetdas_raw                 phasenet das 结果，我跑了几个
        ├── picks_phasenet_das_raw
        ├── picks_phasenet_das_raw_2
        ├── sac_diting
        ├── sac_diting_fig2
        ├── sac_diting_fig_multi
        ├── sac_diting_fig_multi_100Hz
        ├── sac_diting_figxfj_das_diting_eq_148.png
        ├── sac_diting_figxfj_das_diting_eq_153.png
        ├── sac_diting_figxfj_das_diting_eq_160.png
        ├── sac_diting_figxfj_das_diting_eq_168.png
        ├── sac_diting_figxfj_das_diting_eq_175.png
        ├── sac_phasenetdas_fig2
        ├── sac_phasenetdas_fig_2
        ├── tdms_flie.txt
        ├── test_fig
        ├── theory_arrival_cat.csv
        ├── theory_arrival_cat_Pp.csv
        └── xfj_tdms_raw_ditin
├── denoise 降噪模型
│   ├── DeepDenoiser 
│   └── iDAS-self-supervi sed-denoising
├── seg_model 分割模型(本计划用于标注时候辅助)
│   ├── SegNeXt
│   ├── das_event_reorganize_fig
│   └── mmsegmentation
└── src 主要的代码位置
    ├── 202ML0_guangdongCatalog.eqt 广东省地区的地震目录，寇博网络下载
    ├── DasPrep.py 
    ├── DasTools 
    ├── README.md 标注相关文档
    ├── __pycache__
    ├── archive 存档
    ├── beta.ipynb 开发使用的标注工具
    ├── calibrate_das_cable_location_xfj2024.ipynb 标注光纤位置
    ├── cut_eq_win_from_raw_das_data_reorganize.ipynb 切出npy文件
    ├── das2sac.py h5文件转sac的
    ├── fine_location.ipynb
    ├── finetune_val
    ├── get_arrival.ipynb
    ├── h5_to_sac.py
    ├── pick_phase_tool.ipynb
    ├── post.py
    ├── printlog.txt
    ├── view_predict.py
    ├── view_predict_xfj3km.ipynb
    └── view_predict_xfj3km_detect.ipynb
```

## 支持的模型

### 1. SeisT系列 (Seismogram Transformer)
- **SeisT-S/M/L**：不同规模的地震波形Transformer模型
- **用途**：地震检测、震相识别、震级估计等
- **特点**：专为地震数据设计的注意力机制

### 2. EQTransformer
- **用途**：地震检测和震相识别
- **特点**：基于注意力机制的端到端模型
- **参考**：Mousavi et al., 2020, Nature Communications

### 3. PhaseNet
- **用途**：震相识别
- **特点**：经典的卷积神经网络架构
- **参考**：Zhu & Beroza, 2019, Geophysical Research Letters

### 4. MagNet
- **用途**：震级估计
- **特点**：专门设计的震级预测网络

### 5. 基于LLaMA的模型
- **用途**：多任务地震学分析
- **特点**：大语言模型架构应用于地震数据

## 支持的任务

| 任务代码 | 任务名称 | 说明 | 输出格式 |
|---------|---------|------|----------|
| `dpk` | 地震检测和震相识别 | Detection and Phase Picking | 检测概率 + P/S震相时间 |
| `pmp` | P波初动极性分类 | P-motion Polarity | 上升/下降极性 |
| `emg` | 震级估计 | Magnitude Estimation | 震级数值 |
| `baz` | 方位角估计 | Azimuth Estimation | 方位角度 |
| `dis` | 震中距估计 | Distance Estimation | 距离（km）|
| `cls` | 分类任务 | Classification | 类别标签 |

## 数据格式

### 输入数据
- **格式**：HDF5 (.h5/.hdf5)
- **结构**：三分量地震波形数据（Z、N、E）
- **采样率**：100Hz
- **长度**：可配置（默认10000个采样点）

### 数据集结构
```
data/
├── course_data_cls.h5         # 分类任务数据
├── course_data_dpk.h5         # 检测识别数据
├── course_data_emg.h5         # 震级估计数据
├── fewshot_cls_course.hdf5    # 少样本分类数据
├── fewshot_dpk_course.csv     # 少样本检测数据元信息
└── fewshot_emg_course.csv     # 少样本震级数据元信息
```

## 使用方法

### 1. 快速开始

#### 分类任务示例
```bash
cd course_sft_code/course_script
bash cls.sh
```

#### 地震检测和震相识别示例
```bash
cd course_sft_code/course_script
bash dpk.sh
```

#### 震级估计示例
```bash
cd course_sft_code/course_script
bash emg.sh
```

### 2. 自定义训练

#### 基本训练命令
```bash
cd course_sft_code
python main_finetune_DCU.py \
    --mode train_test \
    --downstream-task dpk \
    --model-name llama \
    --pretrained ../ckpt/checkpoint_pt_0100.pth.tar \
    --train_data_dir ../data/course_data_dpk.h5 \
    --epochs 100 \
    --batch-size 32 \
    --base-lr 1e-4
```

#### 参数说明
- `--mode`: 运行模式（train/test/train_test）
- `--downstream-task`: 任务类型（dpk/cls/emg/baz/dis/pmp）
- `--model-name`: 模型名称（llama/seist_m_dpk/eqtransformer等）
- `--pretrained`: 预训练模型路径
- `--train_data_dir`: 训练数据路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--base-lr`: 学习率

### 3. 高级配置

#### 数据增强
```bash
--augmentation true \
--pre-emphasis-rate 0.4 \
--add-gap-rate 0.4 \
--scale-amplitude-rate 0.4
```

#### 分布式训练
```bash
# 单机多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 main_finetune_DCU.py \
    --distributed true \
    [其他参数]
```

#### 学习率调度
```bash
--lr-scheduler cosine \
--warmup-steps 0.02 \
--weight_decay 0.05
```

## 模型配置

### 1. 预训练模型

项目提供了预训练的基础模型：
- `checkpoint_pt_0100.pth.tar`: 100M参数的预训练模型
- 支持多种下游任务的微调

### 2. 超参数配置

#### 不同规模模型的推荐配置

| 模型规模 | 批次大小 | 学习率 | 权重衰减 | 训练轮数 |
|---------|---------|-------|----------|----------|
| 小型 | 64 | 1e-3 | 0.01 | 50 |
| 中型 | 32 | 5e-4 | 0.05 | 100 |
| 大型 | 16 | 1e-4 | 0.1 | 200 |

## 评估和可视化

### 1. 模型评估
```bash
python main_finetune_DCU.py \
    --mode test \
    --checkpoint path/to/model.pth \
    --test_data_dir path/to/test_data.h5
```

### 2. 结果可视化
```bash
--visualize true \
--visualize_save_dir ./visualizations
```

### 3. TensorBoard监控
```bash
tensorboard --logdir ./logs/tensorboard --port 8080
```

## 性能优化

### 1. 内存优化
- 使用梯度累积：`--gradient-accumulation-steps 4`
- 混合精度训练：`--use-amp true`
- 模型并行：适用于大型模型

### 2. 训练加速
- 数据并行：多GPU训练
- 预加载数据：`--pin-memory true`
- 优化数据加载：`--workers 8`

## 常见问题

### 1. 内存不足
```bash
# 减少批次大小
--batch-size 16

# 使用梯度累积
--gradient-accumulation-steps 4

# 启用混合精度
--use-amp true
```

### 2. 训练不收敛
```bash
# 调整学习率
--base-lr 1e-5

# 增加warmup步数
--warmup-steps 0.1

# 使用不同的优化器
--optim adamw
```

### 3. 数据加载错误
- 检查HDF5文件格式
- 确认数据路径正确
- 验证数据集完整性

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 查看LICENSE文件了解详情。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 邮件联系项目维护者

---

**注意**：使用本项目前，请确保您已经正确安装了所有依赖，并且拥有足够的计算资源进行模型训练。建议首先在小规模数据上测试，确认环境配置无误后再进行大规模训练。 