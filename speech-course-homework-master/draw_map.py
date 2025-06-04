import os
import sys
import argparse
import random
import json
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========== 依赖本地模块 ===========
# 确保这些路径和导入与你的项目结构一致
try:
    from data.dataloader import get_dataloader
    from tokenizer.tokenizer import Tokenizer
    from utils.utils import to_device

    # 从你的 model.py 和 subsampling.py 导入必要的构建块
    # Conformer_CTCModel 是你从零开始训练的模型类
    # FineTuneConfig 和 ConformerCTCASRModel 将用于加载微调后的模型
    from model.model import PositionalEncoding, ConformerEncoder, Conformer_CTCModel
    from model.subsampling import Subsampling
    print("Successfully imported user local modules.")
except ImportError as e:
    print(f"Could not import user-defined modules: {e}")
    print("Please ensure data, tokenizer, model, utils directories are in PYTHONPATH or current directory.")
    print("Using placeholder implementations for missing components if any.")
    # --- Fallback Placeholders (如果本地模块导入失败) ---
    class Tokenizer:
        def __init__(self, vocab_file=None): self.word2idx = {'<blk>': 0, '<unk>':1, '<sos>':2, '<eos>':3, 'a':4}; self.idx2word = {v:k for k,v in self.word2idx.items()}
        def size(self): return len(self.word2idx); blk_id = lambda self: self.word2idx['<blk>']
        def __call__(self, text_list): return [[self.word2idx.get(c, 1) for c in t] for t in text_list]
        def decode_to_string(self, id_list): return "".join([self.idx2word.get(i, "<unk>") for i in id_list])
    class Subsampling(nn.Module):
        def __init__(self, in_dim, out_dim, dropout_rate=0.1, subsampling_type=8):
            super().__init__(); print("WARNING: Using placeholder Subsampling"); self.linear = nn.Linear(in_dim, out_dim); self.factor = subsampling_type
        def forward(self, x, x_lens): x=self.linear(x[:,::self.factor,:]); x_lens = x_lens // self.factor; return x, x_lens
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout_rate, max_len=5000):
            super().__init__(); print("WARNING: Using placeholder PositionalEncoding"); self.dropout = nn.Dropout(dropout_rate); self.pe = nn.Parameter(torch.randn(1,max_len,d_model), requires_grad=False)
        def forward(self, x, offset=0): return self.dropout(x + self.pe[:, :x.size(1)]), self.pe[:,:x.size(1)]
    class ConformerEncoder(nn.Module):
        def __init__(self, num_layers, d_model,nhead,ff_ratio,dropout,conv_kernel):
            super().__init__(); print("WARNING: Using placeholder ConformerEncoder"); self.layer = nn.Linear(d_model, d_model)
        def forward(self, x, mask=None): return self.layer(x)
    class Conformer_CTCModel(nn.Module): # 占位符，与你的模型结构一致
        def __init__(self, in_dim, output_size, vocab_size, blank_id, num_layers=6, nhead=8, **kwargs):
            super().__init__(); print("WARNING: Using placeholder Conformer_CTCModel"); self.subsampling = Subsampling(in_dim, output_size); self.pos = PositionalEncoding(output_size,0.1); self.enc = ConformerEncoder(num_layers,output_size,nhead,4,0.1,31); self.fc = nn.Linear(output_size, vocab_size)
        def forward(self, x, x_lens, texts=None, text_lens=None): x,x_lens=self.subsampling(x,x_lens); x,_=self.pos(x,0); x=self.enc(x); logits=self.fc(x); return logits, None, x_lens
    def get_dataloader(wav_scp, pinyin_file, batch_size, tokenizer, shuffle=True, num_workers_override=0): print(f"WARNING: Using PLACEHOLDER get_dataloader."); return None
    def to_device(batch, device): return batch
    # --- End Fallback Placeholders ---

# =========== 配置 (主要用于微调模型加载，但一些参数如 vocab_file 也可共用) ==============
class FineTuneConfig: # 这个类主要用于加载微调模型的配置，但一些路径可以被两个模型共用
    # 数据和词汇表路径
    test_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/test/wav.scp"
    test_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/test/pinyin"
    vocab_file = "./tokenizer/vocab.txt"

    # 模型结构参数 (应与训练时一致)
    mel_input_dim = 80
    n_mels = 80
    conformer_d_model = 256
    conformer_num_layers = 6
    conformer_nhead = 8
    conformer_ff_ratio = 4
    conformer_dropout_rate = 0.1
    conformer_conv_kernel = 31
    conformer_pos_enc_dropout_rate = 0.1
    conformer_subsampling_factor = 8

    # 其他运行时参数
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    seed = 42

    def __init__(self, **kwargs):
        # 设置所有属性为类变量默认
        default_attrs = {k: v for k, v in self.__class__.__dict__.items() if not k.startswith('__') and not callable(v)}
        for k, v in default_attrs.items():
            setattr(self, k, v)
        # 再根据传入的参数覆盖
        for k, v in kwargs.items():
            if hasattr(self, k): # 只设置 FineTuneConfig 中定义的属性
                setattr(self, k, v)
        # 保证 n_mels 和 mel_input_dim 一致
        self.mel_input_dim = getattr(self, "n_mels", self.mel_input_dim)
        self.conformer_subsampling_factor = getattr(self, "conformer_subsampling_factor", 8)


# ========== 微调模型定义 (与 finetune_stage2.py 中的一致) ===========
class ConformerCTCASRModel(nn.Module): # 这是用于微调模型的类
    def __init__(self, config: FineTuneConfig, vocab_size: int, blank_id: int):
        super().__init__()
        self.config = config
        self.subsampling = Subsampling(
            in_dim=config.mel_input_dim,
            out_dim=config.conformer_d_model,
            dropout_rate=0.1, # 假设你的Subsampling接受这个参数
            subsampling_type=config.conformer_subsampling_factor
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config.conformer_d_model,
            dropout_rate=config.conformer_pos_enc_dropout_rate
        )
        self.encoder = ConformerEncoder(
            num_layers=config.conformer_num_layers,
            d_model=config.conformer_d_model,
            nhead=config.conformer_nhead,
            ff_ratio=config.conformer_ff_ratio,
            dropout=config.conformer_dropout_rate,
            conv_kernel=config.conformer_conv_kernel
        )
        self.fc_out = nn.Linear(config.conformer_d_model, vocab_size)

    def forward(self, mel_inputs, mel_input_lengths,
                text_targets=None, text_target_lengths=None): # 评估时可以不传text_*
        subsampled_feat, subsampled_lengths = self.subsampling(mel_inputs, mel_input_lengths)
        encoded_seq, _ = self.positional_encoding(subsampled_feat, 0)
        max_sub_len = encoded_seq.size(1)
        encoder_mask = torch.arange(max_sub_len, device=encoded_seq.device)[None, :] >= subsampled_lengths[:, None]
        encoder_out_seq = self.encoder(encoded_seq, mask=encoder_mask)
        logits = self.fc_out(encoder_out_seq) # (B, L_sub, VocabSize)
        # 评估时通常不需要计算loss，但为了和你的模型定义兼容，这里保留，但返回None
        loss = None
        return logits, loss, subsampled_lengths


# ========== 热力图绘制函数 ==========
def plot_ctc_logits(logits_np, out_path, tokenizer, title="CTC Logits Heatmap", sample_id="sample"):
    """
    绘制CTC Logits的热力图。
    Args:
        logits_np: numpy array, shape (T, V)，应为log_softmax后的概率
        out_path: str, 图片保存路径
        tokenizer: Tokenizer对象，用于获取blank_id和词汇表大小
        title: str, 图像标题
        sample_id: str, 用于文件名
    """
    plt.figure(figsize=(15, 8)) # 调整图像大小以便更好地显示
    plt.imshow(logits_np.T, aspect='auto', origin='lower', interpolation='none', cmap='viridis') #使用viridis colormap
    plt.colorbar(label="Log Probability")
    plt.xlabel("Time Frames")
    plt.ylabel("Vocabulary Index")
    
    blank_id = tokenizer.blk_id()
    vocab_size = tokenizer.size()
    
    # 标记 blank_id 的位置
    if blank_id is not None and 0 <= blank_id < vocab_size:
        plt.axhline(y=blank_id, color='red', linestyle='--', linewidth=1, alpha=0.8, label=f'Blank ID ({blank_id})')
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))


    # 为了可读性，可以只显示部分 y 轴刻度标签
    tick_step = max(1, vocab_size // 20) # 大约显示20个刻度
    plt.yticks(np.arange(0, vocab_size, step=tick_step))
    
    plt.title(f"{title} - {sample_id}")
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # 调整布局给colorbar和legend留空间
    plt.savefig(out_path)
    plt.close()
    print(f"Heatmap saved to {out_path}")

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description="Compare CTC Logits Heatmaps of two models on a single sample.")
    parser.add_argument('--config_file', type=str, default="/data/mingxu/Audio/Assignment_3/ckpt/model_30.pt", help="Path to a JSON config file for FineTuneConfig parameters.")
    parser.add_argument('--ckpt_scratch_path', type=str, default="/data/mingxu/Audio/Assignment_3/ckpt/model_30.pt",
                        help="Path to the checkpoint of the model trained from scratch (e.g., from /data/mingxu/Audio/Assignment_3/ckpt/)")
    parser.add_argument('--ckpt_finetuned_path', type=str,default="/data/mingxu/Audio/Assignment_3/finetune_conformer_checkpoints/finetune_asr_epoch_30.pt",
                        help="Path to the checkpoint of the fine-tuned model (e.g., from /data/mingxu/Audio/Assignment_3/finetune_conformer_checkpoints/)")
    parser.add_argument('--vocab_file', type=str, default="./tokenizer/vocab.txt",
                        help="Path to vocabulary file for Tokenizer")
    parser.add_argument('--test_wav_scp', type=str,
                        help="Path to test wav.scp file (overrides FineTuneConfig)")
    parser.add_argument('--test_pinyin_file', type=str,
                        help="Path to test pinyin file (overrides FineTuneConfig)")
    parser.add_argument('--sample_index', type=int, default=0,
                        help="Index of the sample to pick from the first batch of the test set (0 for first sample).")
    parser.add_argument('--batch_size_eval', type=int, default=4, help="Batch size for dataloader (to get the sample).")
    parser.add_argument('--output_dir', type=str, default="./comparison_heatmaps",
                        help="Directory to save the heatmap images.")
    parser.add_argument('--device', type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_workers', type=int, default=0)


    args = parser.parse_args()

    # --- 1. 基本设置 ---
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_json = json.load(f)
        config = FineTuneConfig(**config_json) # 使用 FineTuneConfig 主要为了模型参数
    else:
        config = FineTuneConfig() # 使用默认值

    # 更新被命令行参数覆盖的配置
    config.device = args.device
    if args.test_wav_scp: config.test_wav_scp = args.test_wav_scp
    if args.test_pinyin_file: config.test_pinyin_file = args.test_pinyin_file
    if args.vocab_file: config.vocab_file = args.vocab_file # 确保 tokenizer 使用正确的 vocab

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. 初始化 Tokenizer 和 Dataloader ---
    tokenizer = Tokenizer(config.vocab_file)
    blank_id = tokenizer.blk_id()

    print(f"Loading test data from: {config.test_wav_scp}, {config.test_pinyin_file}")
    # 确保 get_dataloader 使用 num_workers_override
    test_dataloader = get_dataloader(
        config.test_wav_scp, config.test_pinyin_file,
        args.batch_size_eval, tokenizer, shuffle=False, # shuffle=False for picking a consistent sample
        num_workers_override=args.num_workers
    )

    if test_dataloader is None:
        print("Failed to initialize test_dataloader. Exiting.")
        return

    # --- 3. 获取一个样本 ---
    sample_audio = None
    sample_audio_len = None
    sample_id = "unknown_sample"

    print(f"Fetching sample {args.sample_index} from the first batch of test data...")
    try:
        first_batch = next(iter(test_dataloader))
        first_batch = to_device(first_batch, config.device)

        if args.sample_index < first_batch['audios'].size(0):
            sample_audio = first_batch['audios'][args.sample_index].unsqueeze(0) # (1, L_max, FeatureDim)
            sample_audio_len = first_batch['audio_lens'][args.sample_index].unsqueeze(0) # (1,)
            sample_id = first_batch['ids'][args.sample_index] if 'ids' in first_batch and args.sample_index < len(first_batch['ids']) else f"sample_{args.sample_index}"
            print(f"Selected sample ID: {sample_id}, audio shape: {sample_audio.shape}, length: {sample_audio_len.item()}")
        else:
            print(f"Error: sample_index {args.sample_index} is out of bounds for the first batch size {first_batch['audios'].size(0)}.")
            return
    except StopIteration:
        print("Error: Test dataloader is empty.")
        return
    except Exception as e:
        print(f"Error fetching sample from dataloader: {e}")
        return


    # --- 4. 处理模型1 (从零训练的 Conformer_CTCModel) ---
    print(f"\n--- Evaluating Model 1 (Trained from Scratch) ---")
    print(f"Loading checkpoint: {args.ckpt_scratch_path}")
    try:
        # Conformer_CTCModel (来自 model.model) 的参数通常直接传递
        model_scratch = Conformer_CTCModel(
            in_dim=config.mel_input_dim,          # 应该是 80
            output_size=config.conformer_d_model, # 应该是 256
            vocab_size=tokenizer.size(),
            blank_id=blank_id,
            num_layers=config.conformer_num_layers,
            nhead=config.conformer_nhead
            # 其他 Conformer_CTCModel 可能需要的参数，确保与它训练时一致
        ).to(config.device)

        state_scratch = torch.load(args.ckpt_scratch_path, map_location=config.device)
        # 你的从零训练脚本保存的是 state['model']
        if 'model' in state_scratch:
            model_scratch.load_state_dict(state_scratch['model'])
        elif 'model_state_dict' in state_scratch: # 也尝试一下这个key
             model_scratch.load_state_dict(state_scratch['model_state_dict'])
        else:
            raise KeyError("'model' or 'model_state_dict' key not found in scratch model checkpoint.")
        model_scratch.eval()

        with torch.no_grad():
            logits_scratch, _, out_lens_scratch = model_scratch(sample_audio, sample_audio_len)
        
        # logits_scratch: (1, T_sub, VocabSize)
        # 取第一个样本的 logits，并处理长度
        logits_scratch_sample = logits_scratch[0, :out_lens_scratch[0].item(), :]
        log_probs_scratch = F.log_softmax(logits_scratch_sample, dim=-1).cpu().numpy()

        plot_ctc_logits(
            log_probs_scratch,
            os.path.join(args.output_dir, f"{sample_id}_heatmap_scratch_{os.path.basename(args.ckpt_scratch_path)}.png"),
            tokenizer,
            title="CTC Logits (Model Trained from Scratch)",
            sample_id=sample_id
        )
        print(f"Logits shape from scratch model for sample: {logits_scratch_sample.shape}")

    except Exception as e:
        print(f"Error processing model trained from scratch: {e}")
        import traceback
        traceback.print_exc()


    # --- 5. 处理模型2 (微调后的 ConformerCTCASRModel) ---
    print(f"\n--- Evaluating Model 2 (Fine-tuned) ---")
    print(f"Loading checkpoint: {args.ckpt_finetuned_path}")
    try:
        checkpoint_finetuned = torch.load(args.ckpt_finetuned_path, map_location=config.device)
        
        # 从微调检查点加载配置
        if 'config' in checkpoint_finetuned:
            model_config_dict = checkpoint_finetuned['config']
            # 确保 device 和 vocab_file 使用当前评估脚本的设置或命令行参数
            model_config_dict['device'] = config.device 
            model_config_dict['vocab_file'] = config.vocab_file
            model_finetuned_config = FineTuneConfig(**model_config_dict)
            print(f"  Loaded config from fine-tuned checkpoint: mel_input_dim={model_finetuned_config.mel_input_dim}, conformer_d_model={model_finetuned_config.conformer_d_model}")
        else:
            print("  Warning: 'config' not found in fine-tuned checkpoint. Using default FineTuneConfig.")
            model_finetuned_config = config # Fallback to current script's default config

        # 确保使用正确的维度
        if model_finetuned_config.mel_input_dim != 80:
             print(f"Warning: Fine-tuned model config has mel_input_dim={model_finetuned_config.mel_input_dim}, but data is 80-dim. Overriding to 80 for model init.")
             model_finetuned_config.mel_input_dim = 80
             model_finetuned_config.n_mels = 80


        model_finetuned = ConformerCTCASRModel(
            model_finetuned_config, # 使用从检查点加载或默认的配置
            tokenizer.size(),
            blank_id
        ).to(config.device)

        if 'model_state_dict' not in checkpoint_finetuned:
            raise KeyError("'model_state_dict' not found in fine-tuned model checkpoint.")
        
        model_finetuned.load_state_dict(checkpoint_finetuned['model_state_dict'])
        model_finetuned.eval()

        with torch.no_grad():
            logits_finetuned, _, out_lens_finetuned = model_finetuned(sample_audio, sample_audio_len)

        # logits_finetuned: (1, T_sub, VocabSize)
        logits_finetuned_sample = logits_finetuned[0, :out_lens_finetuned[0].item(), :]
        log_probs_finetuned = F.log_softmax(logits_finetuned_sample, dim=-1).cpu().numpy()

        plot_ctc_logits(
            log_probs_finetuned,
            os.path.join(args.output_dir, f"{sample_id}_heatmap_finetuned_{os.path.basename(args.ckpt_finetuned_path)}.png"),
            tokenizer,
            title="CTC Logits (Fine-tuned Model)",
            sample_id=sample_id
        )
        print(f"Logits shape from fine-tuned model for sample: {logits_finetuned_sample.shape}")

    except Exception as e:
        print(f"Error processing fine-tuned model: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nComparison heatmaps saved in {args.output_dir}")

if __name__ == "__main__":
    main()