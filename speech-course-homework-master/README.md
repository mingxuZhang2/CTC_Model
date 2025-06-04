# 南开大学语音课程大作业 - 基于 Conformer 的语音识别系统

本项目为南开大学语音课程的大作业，旨在实现一个基于 **Conformer** 模型的语音识别 (ASR) 系统。项目采用了**两阶段训练策略**：第一阶段进行预训练，第二阶段进行 ASR 任务的微调。我们通过 Mel 谱重建的可视化结果证明了预训练阶段 **VAE** 结构能有效减少信息损失，并通过 CTC 解码热力图展示了模型的高置信度和稳定性。

---

## 🚀 项目结构

```
speech-course-homework/
├── ctc_logit_heatmaps/ # CTC 解码时 logit 热力图
├── data/               # 数据加载与特征提取
├── model/              # 模型实现 (Transformer & Conformer)
├── pic/                # 辅助图片 (可忽略)
├── reconstruct_image/  # 预训练 Mel 谱重建对比图
├── tokenizer/          # 拼音 Tokenizer
├── utils/              # 辅助工具函数
├── draw.ipynb          # 绘制 Loss/Acc 曲线
├── draw_map.py         # 绘制解码热力图与 Beam Search 实现
├── eval_new.py         # 评估脚本 (计算字错率)
├── evaluate.py         # Baseline 评估脚本
├── *.log               # 训练与评估日志文件
└── training_*.py       # 训练脚本输出文件
```

### 📁 文件夹含义

* **`data/`**: 实现了数据加载器 (`dataloader`) 以及 **Mel 谱特征**的提取方法。
* **`model/`**: 实现了 Baseline 的 **Transformer** 模型（包括位置编码）以及我们主要使用的 **Conformer Encoder** 结构。
* **`pic/`**: 包含一些辅助性图片，可忽略。
* **`reconstruct_image/`**: 存放预训练（Stage 1）过程中，**每个 epoch 随机选取的样本其原始 Mel 谱与重建 Mel 谱的对比图**。通过对比图像细节（能量密度），可以直观地看到我们方法（得益于 VAE）具有较低的信息损失 (Information Loss)。
* **`tokenizer/`**: 实现了**拼音 (Pinyin) 的 Tokenize 方法**，负责将拼音序列转换为模型可处理的 Token ID，以及进行反向解码。
* **`utils/`**: 包含一些通用的辅助函数，例如将数据移动到指定设备 (`to_device`) 等。
* **`ctc_logit_heatmaps/`**: 存放 **CTC 解码过程中的热力图**。这些热力图可以展示模型在解码每个时间步时的输出概率分布，从而说明我们模型解码时具有**较高的置信度和较强的稳定性**。

---

### 📄 文件含义

* **`draw.ipynb`**: 用于**绘制** Baseline 和我们方法在训练过程中的 `training loss` 曲线，以及评估 (`eval`) 过程中的 `accuracy` 曲线等，方便进行性能对比和分析。
* **`draw_map.py`**: 用于**绘制解码过程的热力图**，并实现了 `beam search` 解码算法（注：此处的 `beam search` 实现效率较低，并未在最终评估中重点使用）。
* **`eval_baseline.log`**: 记录了 **Baseline (Transformer) 模型**的评估结果。
* **`eval_conformer.log`**: 记录了**消融实验**中，未使用预训练的 **Conformer 模型**的评估结果。
* **`eval_new.py`**: 用于评估我们最终实现的方法（**预训练 + Conformer**）在各个 checkpoints 上的表现，评估指标为**字错率 (Character Error Rate, CER)**。
* **`eval_pretrain_conformer.log / .txt`**: 记录了我们最终方法的**详细评估结果**。
* **`evaluate.py`**: 评估 Baseline 结果的函数/脚本。
* **`finetune_asr_log`**: 记录了 **Stage 2 (ASR 微调)** 阶段的训练输出日志。
* **`pretrain_cl_recon_log`**: 记录了 **Stage 1 (预训练)** 阶段的训练输出日志。
* **`training_*.py`**: 训练时输出的文件，包含了详细的 loss 等训练过程信息。

---

## ✨ 方法亮点

1.  **Conformer Encoder**: 采用了结合了 CNN 和 Transformer 优势的 Conformer 作为核心 Encoder 结构，有效捕捉语音信号的局部和全局依赖。
2.  **VAE 预训练**: 在 Stage 1 引入 VAE 进行 Mel 谱重建任务，显著降低了特征提取过程中的信息损失，为下游任务提供了更高质量的表征。
3.  **CTC 解码与可视化**: 使用 CTC 进行端到端的语音识别，并通过热力图可视化解码过程，验证模型的性能和稳定性。
4.  **详尽的实验记录**: 提供了多种模型（Baseline, Conformer w/o pretrain, Conformer w/ pretrain）的训练和评估日志，方便复现和对比。