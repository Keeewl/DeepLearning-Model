# DeepLearning-Model
This is a repository for implementing deep learning models.

## Installation
Conda env:
```bash
# Create conda env:
conda create -n dl-model python=3.10 -y
conda activate dl-model

# Install pytorch for NVIDIA GeForce RTX 5090:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Or you can directly setup conda environment:
conda env create -f environment.yml

# Check env:
python utils/env_check.py
```

## ToDo List
Modules:
- [] MLP
- [] FFN
- [] CNN
- [] RNN  
- [] Attention

Models:
- [] Transformer
- [] Vision Transformer
- [] Diffusion
- [] Conditioning

## Project Structure
```bash
DeepLearning-Model/
├── dl/                           # ✅主包（DeepLearning-Model）
│   │
│   ├── modules/                  # ✅你手写的“可读实现”（核心）
│   │   ├── __init__.py
│   │   ├── mlp.py                # Linear/MLP/FFN blocks（含shape注释）
│   │   ├── norm.py               # LN/RMSNorm/BN（写清eps/affine）
│   │   ├── activations.py        # gelu/silu/relu等（可选）
│   │   ├── attention.py          # MHA/CrossAttn + mask utils
│   │   ├── embeddings.py         # pos/time embeddings（sin/cos, learned）
│   │   ├── resnet_blocks.py      # BasicBlock/Bottleneck（学习用）
│   │   ├── vit_blocks.py         # PatchEmbed/EncoderBlock（学习用）
│   │   ├── diffusion_blocks.py   # time-MLP/AdaLN/FiLM（学习用）
│   │   └── init.py               # xavier/kaiming/truncnorm等初始化
│   │
│   ├── reference/                # ✅“库实现对照组”（torch/timm等的薄封装）
│   │   ├── __init__.py
│   │   ├── mlp_ref.py            # nn.Sequential/nn.Linear版
│   │   ├── norm_ref.py           # nn.LayerNorm/BatchNorm版
│   │   ├── attention_ref.py      # nn.MultiheadAttention 或等价实现
│   │   ├── vit_ref.py            # 可选：timm ViT 对照（薄封装）
│   │   └── diffusion_ref.py      # 可选：用成熟实现/简化版对照
│   │
│   ├── models/                   # ✅由 modules 组装的“学习版完整模型”
│   │   ├── __init__.py
│   │   ├── mlp_regressor.py      # fit sinx / regression
│   │   ├── mlp_classifier.py     # toy分类 / MNIST
│   │   ├── resnet_tiny.py        # CIFAR级别小ResNet
│   │   ├── transformer_tiny.py   # toy序列任务（copy/reverse）
│   │   ├── vit_tiny.py           # 小ViT（CIFAR/CatsDogs级别）
│   │   ├── ddpm_toy.py           # 2D toy diffusion（GMM/spiral）
│   │   └── ddpm_seq_toy.py       # toy序列diffusion（贴近motion直觉）
│   │
│   ├── checks/                   # ✅关键：正确性/对齐/数值检查（不追指标）
│   │   ├── __init__.py
│   │   ├── common.py             # compare_tensors/grad_check/seed
│   │   ├── test_mlp_equiv.py     # 手写 vs reference：forward+backward
│   │   ├── test_norm_equiv.py
│   │   ├── test_attention_mask.py# mask广播/causal正确性
│   │   ├── test_vit_shapes.py    # patch/pos/cls拼接shape流
│   │   └── test_ddpm_math.py     # q(x_t|x0) closed-form数值对齐
│   │
│   ├── micro_tasks/              # ✅“几分钟闭环”的训练任务（学习训练过程）
│   │   ├── __init__.py
│   │   ├── fit_sinx.py           # 回归：loss曲线/初始化/优化器对比
│   │   ├── mnist_mlp.py          # 小分类：过拟合/正则化/seed复现
│   │   ├── cifar_resnet.py       # 小CNN任务（不追SOTA）
│   │   ├── cifar_vit.py          # 小ViT任务（看shape/训练稳定）
│   │   ├── toy_seq_copy.py       # transformer sanity
│   │   ├── toy_ddpm_2d.py        # diffusion 2D 生成 + 采样可视化
│   │   └── toy_ddpm_seq.py       # 序列diffusion toy（更贴近motion）
│   │
│   ├── engine/                   # ✅轻量训练引擎（只服务 micro_tasks）
│   │   ├── __init__.py
│   │   ├── trainer.py            # 单机单卡：amp/clip/accum/ckpt/log
│   │   ├── loops.py              # train_one_epoch/eval_one_epoch
│   │   ├── optim.py              # build_optimizer/build_scheduler
│   │   ├── logger.py             # print + tensorboard（可选）
│   │   ├── checkpoint.py         # save/load last/best
│   │   └── seed.py               # 固定随机种子/可复现
│   │
│   └── utils/                    # ✅通用小工具（学习用）
│       ├── __init__.py
│       ├── shapes.py             # shape断言/pretty print（强烈建议）
│       ├── profile.py            # 简单耗时/显存统计（可选）
│       ├── plot.py               # 画loss曲线/分布图（可选）
│       └── config.py             # 轻量配置（argparse/dataclass）
│
├── scripts/                      # ✅统一入口（让使用体验像“工具箱”）
│   ├── run_check.py              # 运行 checks：python scripts/run_check.py test_mlp_equiv
│   ├── run_task.py               # 运行 micro_tasks：python scripts/run_task.py fit_sinx
│   └── list_things.py            # 列出可运行的checks/tasks
│
├── tests/                        # ✅pytest层（可选，但建议有）
│   ├── test_imports.py
│   └── test_smoke.py
│
├── notebooks/                    # 可选：交互式推导/可视化（不影响主代码）
│   ├── attention_playground.ipynb
│   └── ddpm_derivation.ipynb
│
├── outputs/                      # 运行产物（自动生成，不进git）
│   ├── micro_tasks/
│   └── checks/
│
└── data/                         # 小数据/缓存（不进git）
    ├── cache/
    └── tiny_samples/
```