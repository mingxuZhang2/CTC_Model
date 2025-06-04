import os
import sys
import argparse
import random
import json
import warnings
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
# from torchaudio.transforms import MelSpectrogram # 如果SimpleMelDecoder等需要，但评估脚本通常不需要
import numpy as np
from tqdm import tqdm
import time

# --- 用户本地模块的导入 ---
# 确保这些导入路径与你的项目结构一致，并且能加载到正确的模块
try:
    from data.dataloader import get_dataloader
    from tokenizer.tokenizer import Tokenizer
    from utils.utils import to_device
    # 以下是构建 ConformerCTCASRModel 所需的组件
    from model.model import PositionalEncoding, ConformerEncoder # 来自你的 model/model.py
    from model.subsampling import Subsampling                   # 来自你的 model/subsampling.py
    print("Successfully imported user local modules for evaluation.")
except ImportError as e:
    print(f"Could not import user-defined modules for evaluation: {e}")
    print("Ensure paths are correct and all necessary model components are available.")
    print("Using placeholder implementations for missing components if any.")
    # 占位符 (仅在上述导入失败时作为后备，理想情况下不应执行到这里)
    class Tokenizer:
        def __init__(self, vocab_file=None): self.word2idx = {'<blk>': 0, '<unk>':1, '<sos>':2, '<eos>':3, 'a':4}; self.idx2word = {v:k for k,v in self.word2idx.items()}
        def size(self): return len(self.word2idx)
        def blk_id(self): return self.word2idx['<blk>']
        def __call__(self, text_list): return [[self.word2idx.get(c, 1) for c in t] for t in text_list]
        def decode_to_string(self, id_list): return "".join([self.idx2word.get(i, "<unk>") for i in id_list])
    class Subsampling(nn.Module):
        def __init__(self, in_dim, out_dim, dropout_rate=0.1, subsampling_type=8):
            super().__init__(); print("WARNING: Using placeholder Subsampling for eval")
            self.linear = nn.Linear(in_dim, out_dim); self.factor = subsampling_type
        def forward(self, x, x_lens): x=self.linear(x[:,::self.factor,:]); x_lens = x_lens // self.factor; return x, x_lens
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout_rate, max_len=5000):
            super().__init__(); print("WARNING: Using placeholder PositionalEncoding for eval")
            self.dropout = nn.Dropout(dropout_rate); self.pe = nn.Parameter(torch.randn(1,max_len,d_model), requires_grad=False)
        def forward(self, x, offset=0): return self.dropout(x + self.pe[:, :x.size(1)]), self.pe[:,:x.size(1)]
    class ConformerEncoder(nn.Module):
        def __init__(self, num_layers, d_model,nhead,ff_ratio,dropout,conv_kernel):
            super().__init__(); print("WARNING: Using placeholder ConformerEncoder for eval")
            self.layer = nn.Linear(d_model, d_model)
        def forward(self, x, mask=None): return self.layer(x)
    def get_dataloader(wav_scp, pinyin_file, batch_size, tokenizer, shuffle=True, num_workers_override=0):
        print(f"WARNING: Using PLACEHOLDER get_dataloader for eval. Feature dim will be 80."); return None
    def to_device(batch, device): return batch


# --- 与 finetune_stage2.py 中相同的 FineTuneConfig 类 ---
# (或者从一个共享的config文件中导入)
class FineTuneConfig:
    train_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/train/wav.scp"
    train_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/train/pinyin"
    test_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/test/wav.scp"
    test_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/test/pinyin"
    mel_input_dim = 80
    n_mels = 80
    conformer_d_model = 256; conformer_num_layers = 6; conformer_nhead = 8
    conformer_ff_ratio = 4; conformer_dropout_rate = 0.1; conformer_conv_kernel = 31
    conformer_pos_enc_dropout_rate = 0.1; conformer_subsampling_factor = 8
    epochs = 20; batch_size = 32; learning_rate = 5e-5; encoder_learning_rate = 1e-5
    freeze_encoder_epochs = 0; grad_clip = 5.0
    device = "cuda:2" if torch.cuda.is_available() else "cpu"; seed = 42; num_workers = 0
    pretrained_checkpoint_path = "" # 在评估时不需要这个
    finetune_checkpoint_dir = "./finetune_conformer_checkpoints" # 默认微调模型保存位置
    log_file = "./evaluate_finetuned_asr_log.txt" # 当前脚本的日志
    save_interval = 1

    def __init__(self, **kwargs): # 允许通过kwargs覆盖，主要用于从checkpoint加载配置
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # 确保一致性，即使kwargs中包含这些
        if 'n_mels' in kwargs : self.mel_input_dim = self.n_mels
        else: self.mel_input_dim = type(self).n_mels # 使用类变量默认值

        if 'conformer_subsampling_factor' not in kwargs: # 确保有默认值
             self.conformer_subsampling_factor = type(self).conformer_subsampling_factor

def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

# --- 与 finetune_stage2.py 中相同的 ASR 模型定义 ---
class ConformerCTCASRModel(nn.Module):
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
        # 在评估时，如果只做推理，ctc_loss不是必须的，但如果forward计算了loss就需要
        self.ctc_loss_fn = torch.nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(self, mel_inputs, mel_input_lengths, 
                text_targets=None, text_target_lengths=None): # text_* 设为可选，便于推理
        subsampled_feat, subsampled_lengths = self.subsampling(mel_inputs, mel_input_lengths)
        encoded_seq, _ = self.positional_encoding(subsampled_feat, 0)
        max_sub_len = encoded_seq.size(1)
        encoder_mask = torch.arange(max_sub_len, device=encoded_seq.device)[None, :] >= subsampled_lengths[:, None]
        encoder_out_seq = self.encoder(encoded_seq, mask=encoder_mask)
        logits = self.fc_out(encoder_out_seq) # (B, L_sub, VocabSize)
        
        loss = None
        if text_targets is not None and text_target_lengths is not None:
            log_probs_for_ctc = F.log_softmax(logits, dim=2).permute(1, 0, 2)
            
            # 处理CTC长度问题 (与微调脚本中类似)
            if torch.any(subsampled_lengths < text_target_lengths):
                clamped_text_target_lengths = torch.min(text_target_lengths, subsampled_lengths)
                valid_indices = clamped_text_target_lengths > 0
                if not torch.all(valid_indices):
                    if not torch.any(valid_indices):
                         loss = torch.tensor(0.0, device=log_probs_for_ctc.device) # No valid samples
                    else: # Filter samples
                        log_probs_for_ctc = log_probs_for_ctc[:, valid_indices, :]
                        current_text_targets = text_targets[valid_indices]
                        current_subsampled_lengths = subsampled_lengths[valid_indices]
                        current_text_target_lengths = clamped_text_target_lengths[valid_indices]
                        loss = self.ctc_loss_fn(log_probs_for_ctc, current_text_targets, current_subsampled_lengths, current_text_target_lengths)
                else: # All samples fine after clamping
                    loss = self.ctc_loss_fn(log_probs_for_ctc, text_targets, subsampled_lengths, clamped_text_target_lengths)
            else:
                loss = self.ctc_loss_fn(log_probs_for_ctc, text_targets, subsampled_lengths, text_target_lengths)

        return logits, loss, subsampled_lengths # 返回logits用于解码

# --- 解码和评估函数 (与你提供的类似) ---
def greedy_search(ctc_logits: torch.tensor, encoder_out_lens: torch.tensor, blank_id: int):
    """
    Args:
        ctc_logits: [B, T, V] raw logits from the model
        encoder_out_lens: [B]
        blank_id: int, CTC blank token id
    Returns:
        hyps: List[List[int]]，每句话的预测id序列
    """
    # 首先应用 log_softmax 得到 log_probs，然后再 topk
    # 如果ctc_logits已经是log_probs，则不需要 F.log_softmax
    # 假设 ctc_logits 是原始 logits
    ctc_log_probs = F.log_softmax(ctc_logits, dim=2)
    batch_size, maxlen = ctc_log_probs.size()[:2]
    
    # topk 在 log_probs 上操作通常更稳定
    _, topk_index = ctc_log_probs.topk(1, dim=2) # 取概率最大的索引
    topk_index = topk_index.view(batch_size, maxlen)
    # encoder_out_lens = encoder_out_lens.view(-1).tolist() # 确保是一维列表

    hyps = []
    for i in range(batch_size): # 使用 batch_size 迭代
        actual_len = encoder_out_lens[i].item() # 获取当前样本的真实输出长度
        ids = topk_index[i, :actual_len].tolist()
        hyp = []
        prev = -1 # 初始化为与任何有效id不同的值
        for idx in ids:
            if idx != blank_id and idx != prev:
                hyp.append(idx)
            # 只有当 idx 不是 blank 时，它才应该成为下一个 prev (用于去重)
            # 如果 idx 是 blank，prev 应该保持不变，以便下一个非 blank 字符可以与 blank 前的字符比较
            if idx != blank_id:
                prev = idx
        hyps.append(hyp)
    return hyps

def cer_stat(pre, gt):
    m, n = len(pre), len(gt)
    dp = [[(0, 0, 0, 0) for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1): dp[i][0] = (i, 0, 0, i)
    for j in range(1, n + 1): dp[0][j] = (j, 0, j, 0)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pre[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                rep = dp[i - 1][j - 1]
                dele = dp[i][j - 1]
                ins = dp[i - 1][j]
                candidates = [
                    (rep[0] + 1, rep[1] + 1, rep[2], rep[3]),
                    (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                    (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])
    min_dist, S, D, I = dp[m][n]
    N = n
    cer_val = (S + D + I) / (N + 1e-8) if N > 0 else ( (S+D+I) / (m + 1e-8) if m > 0 else 0.0) # Handle empty ground truth
    return cer_val, S, D, I, N

# --- 主评估逻辑 ---
def main_evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Conformer ASR Model")
    parser.add_argument('--ckpt_dir', type=str, default=FineTuneConfig.finetune_checkpoint_dir,
                        help="Directory containing fine-tuned checkpoints")
    parser.add_argument('--test_wav_scp', type=str, default=FineTuneConfig.test_wav_scp)
    parser.add_argument('--test_pinyin_file', type=str, default=FineTuneConfig.test_pinyin_file)
    parser.add_argument('--vocab_file', type=str, default="./tokenizer/vocab.txt", # 与你Tokenizer定义一致
                        help="Path to vocabulary file for Tokenizer")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for evaluation")
    parser.add_argument('--device', type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--log_file', type=str, default="./evaluate_finetuned_asr.log",
                        help="Log file to save CER results")
    parser.add_argument('--num_workers', type=int, default=0)


    args = parser.parse_args()
    set_seed(42) # 保持与训练一致的随机性（如果模型中有dropout等）

    tokenizer = Tokenizer(args.vocab_file) # 使用你实际的Tokenizer
    blank_id = tokenizer.blk_id()

    # 加载测试数据
    # 注意：get_dataloader 现在也接收 num_workers_override
    test_dataloader = get_dataloader(
        args.test_wav_scp, args.test_pinyin_file,
        args.batch_size, tokenizer, shuffle=False,
    )

    if not os.path.exists(args.ckpt_dir):
        print(f"Checkpoint directory {args.ckpt_dir} does not exist.")
        return

    ckpt_list = [os.path.join(args.ckpt_dir, x) for x in os.listdir(args.ckpt_dir) if x.endswith(".pt")]
    ckpt_list.sort()

    print(f"Found {len(ckpt_list)} checkpoints in {args.ckpt_dir}")

    with open(args.log_file, "w", encoding="utf-8") as f_log:
        f_log.write("ckpt\tCER\tS\tD\tI\tN\n") # 表头

        for ckpt_path in ckpt_list:
            print(f"\nEvaluating checkpoint: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location=args.device)
            except Exception as e:
                print(f"Error loading checkpoint {ckpt_path}: {e}")
                continue

            if 'config' not in checkpoint:
                print(f"Warning: 'config' not found in checkpoint {ckpt_path}. Using default FineTuneConfig.")
                eval_config = FineTuneConfig(device=args.device) # 使用命令行或默认device
            else:
                # 从checkpoint中加载配置来实例化FineTuneConfig
                # 这确保模型参数与训练时完全匹配
                saved_config_dict = checkpoint['config']
                # 更新device参数，以防checkpoint中保存的device与当前评估device不同
                saved_config_dict['device'] = args.device 
                eval_config = FineTuneConfig(**saved_config_dict)
            
            # 打印当前评估使用的关键配置
            print(f"  Using config for this checkpoint: mel_input_dim={eval_config.mel_input_dim}, conformer_d_model={eval_config.conformer_d_model}, num_layers={eval_config.conformer_num_layers}")

            model = ConformerCTCASRModel(
                eval_config, # 使用从检查点（或默认）加载的配置
                tokenizer.size(),
                blank_id
            ).to(args.device)

            if 'model_state_dict' not in checkpoint:
                print(f"Error: 'model_state_dict' not found in checkpoint {ckpt_path}.")
                continue
            
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"Error loading state_dict for {ckpt_path}: {e}")
                print("This might happen if model architecture changed or config mismatch.")
                continue
                
            model.eval()

            total_S, total_D, total_I, total_N = 0, 0, 0, 0
            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc=f"Eval {os.path.basename(ckpt_path)}", ncols=120):
                    batch = to_device(batch, args.device)
                    audios = batch['audios']               # (B, L_max, 80)
                    audio_lens = batch['audio_lens']       # (B,)
                    texts_gt = batch['texts']            # (B, T_text_max) ground truth token ids
                    text_lens_gt = batch['text_lens']    # (B,) ground truth text lengths
                    
                    # 模型前向传播，在评估时不需要传入GT文本 (除非你的模型forward强制要求)
                    # ConformerCTCASRModel的forward已修改为text_targets和text_target_lengths可选
                    logits, _, encoder_out_lens = model(audios, audio_lens) 
                    # logits: (B, L_sub, VocabSize)
                    # encoder_out_lens: (B,) subsampled audio lengths

                    pred_ids_batch = greedy_search(logits.cpu(), encoder_out_lens.cpu(), blank_id)

                    for i in range(len(pred_ids_batch)):
                        pred_ids = pred_ids_batch[i]
                        # 获取真实的GT token id序列
                        tgt_ids = texts_gt[i][:text_lens_gt[i]].cpu().tolist()
                        
                        _, S, D, I, N = cer_stat(pred_ids, tgt_ids)
                        total_S += S
                        total_D += D
                        total_I += I
                        total_N += N
            
            cer = (total_S + total_D + total_I) / (total_N + 1e-8) if total_N > 0 else 0.0
            result_line = f"{os.path.basename(ckpt_path)}\t{cer:.6f}\t{total_S}\t{total_D}\t{total_I}\t{total_N}\n"
            print(result_line.strip())
            f_log.write(result_line)
            f_log.flush() # 确保及时写入

if __name__ == "__main__":
    main_evaluate()

