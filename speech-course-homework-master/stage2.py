import os
import sys
import argparse
import random
import json
from typing import Union
import warnings
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram
import numpy as np
# 注意：transformers 不是直接用于ASR模型本身，但在预训练中用于文本编码器，微调时不需要
# from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

# --- 用户本地模块的导入 ---
# 假设这些模块与预训练时使用的相同
try:
    from data.dataloader import get_dataloader # 你的数据加载器
    from tokenizer.tokenizer import Tokenizer  # 你的分词器
    from utils.utils import to_device         # 你的工具函数
    # 从 model.model 导入 ASR 模型结构或其组件
    # 如果你的 Conformer_CTCModel 在 model.model.py 中，可以直接导入
    # 否则，需要确保 Subsampling, PositionalEncoding, ConformerEncoder 可用
    from model.model import PositionalEncoding, ConformerEncoder # 来自你提供的 model.py
    from model.subsampling import Subsampling # 来自你提供的 subsampling.py
    print("Successfully imported user local modules (dataloader, tokenizer, utils, model components).")
except ImportError as e:
    print(f"Could not import user-defined modules: {e}")
    print("Ensure the script is run from the correct project directory or sys.path is configured.")
    print("Using placeholder implementations for missing ASR model components.")
    # 为 ASR 模型组件提供占位符（如果导入失败）
    class Tokenizer:
        def __init__(self): self.word2idx = {'<blk>': 0, '<unk>':1, 'a':2, 'b':3, '<sos>':4, '<eos>':5}; self.idx2word = {v:k for k,v in self.word2idx.items()}
        def size(self): return len(self.word2idx)
        def blk_id(self): return self.word2idx['<blk>']
        def encode(self, text_list): return [[self.word2idx.get(c, 1) for c in text] for text in text_list]
        def __call__(self, text_list): return self.encode(text_list) # 模拟你的tokenizer
        def decode(self, id_list): return [str(self.idx2word.get(int(i), '<unk>')) for i in id_list if isinstance(i, (int, float))]
        def decode_to_string(self, id_list): return "".join(self.decode(id_list))

    def get_dataloader(wav_scp, pinyin_file, batch_size, tokenizer, shuffle=True, num_workers_override=0):
        print(f"WARNING: Using PLACEHOLDER get_dataloader for ASR. Feature dim will be 80. Output shape (B, L, D).")
        class DummyDataset(Dataset):
            def __init__(self, feature_dim=80):
                self.data = []
                self.feature_dim = feature_dim
                for _ in range(100):
                    audio_len = random.randint(100,300); text_len = random.randint(5,15)
                    self.data.append({ 'audios': torch.randn(audio_len, self.feature_dim), 'audio_lens': torch.tensor([audio_len]), 'texts': torch.randint(0, tokenizer.size(), (text_len,)), 'text_lens': torch.tensor([text_len]) })
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return "dummy_id", self.data[idx]['audios'], self.data[idx]['texts'] # 模拟你的 BZNSYP 输出
        
        def placeholder_collate_fn(batch):
            ids = [item[0] for item in batch]
            audios = [item[1] for item in batch] 
            texts = [item[2] for item in batch]
            audio_lens = torch.tensor([s.size(0) for s in audios], dtype=torch.long)
            text_lens = torch.tensor([len(s) for s in texts], dtype=torch.long)
            audios_padded = nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
            texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.blk_id())
            return {'ids': ids, 'audios': audios_padded, 'audio_lens': audio_lens, 'texts': texts_padded, 'text_lens': text_lens}
        return DataLoader(DummyDataset(feature_dim=80), batch_size=batch_size, collate_fn=placeholder_collate_fn, num_workers=num_workers_override)

    def to_device(batch, device):
        if isinstance(batch, dict): return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch.to(device) if isinstance(batch, torch.Tensor) else batch

    class Subsampling(nn.Module): 
        def __init__(self, in_dim, out_dim, subsampling_type=8, dropout_rate=0.1): # 匹配你的subsampling.py
            super().__init__(); print("WARNING: Using placeholder Subsampling (Conv2dSubsampling8 style)")
            # 这是对你 Conv2dSubsampling8 的简化模拟，实际应使用你的版本
            self.conv = nn.Sequential(
                nn.Conv2d(1, out_dim, 3, 2), nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 3, 2), nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 3, 2), nn.ReLU(),
            )
            # 计算线性层的输入维度，这需要精确匹配你的 Conv2dSubsampling8
            # 假设 in_dim = 80, odim = 256
            # f0 = 80
            # f1 = (80 - 1) // 2 - 1 = 39 - 1 = 38 (assuming padding=0 for formula, but your code has padding)
            # Corrected based on your Conv2dSubsampling8:
            # idim (80) -> f_after_conv1 = (idim - 3)//2 + 1 = (80-3)//2 + 1 = 77//2 + 1 = 38+1 = 39
            # f_after_conv2 = (39-3)//2 + 1 = 36//2 + 1 = 18+1 = 19
            # f_after_conv3 = (19-3)//2 + 1 = 16//2+1 = 8+1 = 9
            linear_in_features = out_dim * 9 # 假设 in_dim=80
            if in_dim != 80: print(f"Warning: Placeholder Subsampling linear_in_features calculation assumes in_dim=80, but got {in_dim}")
            self.linear = nn.Linear(linear_in_features, out_dim)

        def forward(self, x, x_lens): # x is (B, Time, FeatureDim)
            x = x.unsqueeze(1)  # (B, 1, Time, FeatureDim)
            x = x.permute(0, 1, 3, 2) # (B, 1, FeatureDim, Time) for Conv2d
            x = self.conv(x) # (B, out_dim, NewFeatureDim, NewTime)
            b, c, f, t = x.size() # c is out_dim, f is new feature dim (e.g. 9)
            x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f) # (B, NewTime, out_dim * NewFeatureDim)
            x = self.linear(x) # (B, NewTime, out_dim)
            
            # Update lengths (simplified for placeholder)
            for _ in range(3): x_lens = (x_lens - 1) // 2 + 1 # Approximate for 3 conv layers stride 2
            max_len_out = x_lens.max().item() if x_lens.numel() > 0 else 0
            x = x[:, :max_len_out, :] if max_len_out > 0 else x[:, :0, :]
            return x, x_lens

    class PositionalEncoding(nn.Module): 
        def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
            super().__init__(); self.d_model = d_model; self.xscale = math.sqrt(self.d_model)
            self.dropout = nn.Dropout(p=dropout_rate); self.pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
            self.pe[:, 0::2] = torch.sin(position * div_term); self.pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = self.pe.unsqueeze(0); self.register_buffer("_pe", self.pe) # Use different name if self.pe is used
        def forward(self, x: torch.Tensor, offset:Union[int, torch.Tensor]=0): # x is (B, Time, Dim)
            pos_emb = self._pe[:, offset:(offset + x.size(1))] if isinstance(offset, int) else self._pe[:, offset.item():(offset.item() + x.size(1))]
            x = x * self.xscale + pos_emb; return self.dropout(x), self.dropout(pos_emb)


    class ConformerEncoder(nn.Module):
        def __init__(self, num_layers, d_model, nhead, ff_ratio=4, dropout=0.1, conv_kernel=31, **kwargs):
            super().__init__(); print("WARNING: Using placeholder ConformerEncoder (single Linear layer)")
            self.fc = nn.Linear(d_model, d_model) # Simplified
        def forward(self, x, mask=None): return self.fc(x)


# --- ASR 模型定义 (基于你的 Conformer_CTCModel 结构) ---
class ConformerCTCASRModel(nn.Module):
    def __init__(self, config, vocab_size, blank_id):
        super().__init__()
        self.config = config
        # 使用与预训练时相同的组件和配置
        self.subsampling = Subsampling(
            in_dim=config.mel_input_dim, # 应该是80
            out_dim=config.conformer_d_model,
            dropout_rate=0.1, # 你的 Subsampling 构造函数接受 dropout_rate
            subsampling_type=config.conformer_subsampling_factor # 你的 Subsampling 构造函数接受 subsampling_type
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
        self.ctc_loss = torch.nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True) # 使用 mean reduction

    def forward(self, mel_inputs, mel_input_lengths, text_targets, text_target_lengths):
        # mel_inputs: (B, L_max, FeatureDim=80)
        # mel_input_lengths: (B,)
        # text_targets: (B, T_max_text)
        # text_target_lengths: (B,)

        # 1. Subsampling
        # 你的 Subsampling (Conv2dSubsamplingX) 期望输入 (B, Time, FeatureDim)
        subsampled_feat, subsampled_lengths = self.subsampling(mel_inputs, mel_input_lengths)
        # subsampled_feat: (B, L_sub, D_model)
        # subsampled_lengths: (B,)

        # 2. Positional Encoding
        encoded_seq, _ = self.positional_encoding(subsampled_feat, 0) # 你的 PositionalEncoding 期望 (B, Time, Dim)

        # 3. Conformer Encoder
        # 创建 key_padding_mask for ConformerEncoder
        # True 表示该位置被遮盖
        max_sub_len = encoded_seq.size(1)
        encoder_mask = torch.arange(max_sub_len, device=encoded_seq.device)[None, :] >= subsampled_lengths[:, None]
        
        encoder_out_seq = self.encoder(encoded_seq, mask=encoder_mask) # 你的 ConformerEncoder 接受 mask (key_padding_mask)
        # encoder_out_seq: (B, L_sub, D_model)

        # 4. Output Layer for CTC
        logits = self.fc_out(encoder_out_seq)  # (B, L_sub, VocabSize)
        
        # 5. Prepare for CTC Loss
        log_probs = F.log_softmax(logits, dim=2) # (B, L_sub, VocabSize)
        log_probs_for_ctc = log_probs.permute(1, 0, 2) # (L_sub, B, VocabSize) - CTC期望时间维度在前

        # CTC loss 计算
        # text_targets 已经是padding好的，text_target_lengths 是真实长度
        # subsampled_lengths 是 subsampling 后的音频帧长度
        
        # 确保 input_lengths >= target_lengths for CTC
        # 有时由于下采样，可能出现 subsampled_lengths < text_target_lengths 的情况
        # 这里可以加一个检查，或者在数据预处理时确保
        if torch.any(subsampled_lengths < text_target_lengths):
            # print(f"Warning: Some input lengths ({subsampled_lengths.min().item()}) are shorter than target lengths ({text_target_lengths.min().item()}) after subsampling. This can lead to CTC errors or NaN loss.")
            # One simple fix: clamp target_lengths to be at most subsampled_lengths
            # This is a HACK, better to ensure data quality or handle this in dataloader/preprocessing
            clamped_text_target_lengths = torch.min(text_target_lengths, subsampled_lengths)
            # Filter out samples where new target length is 0
            valid_indices = clamped_text_target_lengths > 0
            if not torch.all(valid_indices): # If any target length became 0
                # print(f"Warning: Some target lengths became 0 after clamping. Skipping these samples for loss calculation.")
                if not torch.any(valid_indices): # If ALL samples are invalid
                     return logits, torch.tensor(0.0, device=log_probs.device, requires_grad=True), subsampled_lengths

                log_probs_for_ctc = log_probs_for_ctc[:, valid_indices, :]
                text_targets = text_targets[valid_indices]
                subsampled_lengths_ctc = subsampled_lengths[valid_indices]
                text_target_lengths_ctc = clamped_text_target_lengths[valid_indices]
            else:
                subsampled_lengths_ctc = subsampled_lengths
                text_target_lengths_ctc = clamped_text_target_lengths
        else:
            subsampled_lengths_ctc = subsampled_lengths
            text_target_lengths_ctc = text_target_lengths


        loss = self.ctc_loss(log_probs_for_ctc, text_targets, subsampled_lengths_ctc, text_target_lengths_ctc)
        
        return logits, loss, subsampled_lengths

    def load_pretrained_acoustic_encoder(self, checkpoint_path: str, device: str):
        """加载预训练的声学编码器权重"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Pretrained checkpoint not found at {checkpoint_path}. Training from scratch.")
            return
        
        print(f"Loading pretrained acoustic encoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        loaded_something = False
        if 'subsampling_state_dict' in checkpoint and checkpoint['subsampling_state_dict']:
            self.subsampling.load_state_dict(checkpoint['subsampling_state_dict'])
            print("  Loaded subsampling weights.")
            loaded_something = True
        else:
            print("  Warning: 'subsampling_state_dict' not found or empty in checkpoint.")

        if 'positional_encoding_state_dict' in checkpoint and checkpoint['positional_encoding_state_dict']:
            # 你的PositionalEncoding的pe是buffer，可能需要特殊处理，但通常load_state_dict能处理
            # 如果你的PositionalEncoding的pe不是通过register_buffer动态创建的，而是固定大小的，
            # 且预训练和微调的max_len不同，可能会有问题。
            # 假设你的PositionalEncoding实现与预训练时一致。
            try:
                self.positional_encoding.load_state_dict(checkpoint['positional_encoding_state_dict'])
                print("  Loaded positional_encoding weights.")
                loaded_something = True
            except RuntimeError as e:
                 print(f"  Warning: Error loading positional_encoding weights: {e}. This might be due to PE buffer size mismatch if max_len changed.")
        else:
            print("  Warning: 'positional_encoding_state_dict' not found or empty in checkpoint.")

        if 'acoustic_encoder_state_dict' in checkpoint and checkpoint['acoustic_encoder_state_dict']:
            # 预训练时叫 acoustic_encoder, ASR模型中叫 encoder
            self.encoder.load_state_dict(checkpoint['acoustic_encoder_state_dict'])
            print("  Loaded acoustic_encoder (ConformerEncoder) weights.")
            loaded_something = True
        else:
            print("  Warning: 'acoustic_encoder_state_dict' not found or empty in checkpoint.")
        
        if not loaded_something:
            print("Warning: No weights were loaded from the checkpoint for the acoustic encoder.")
        else:
            print("Pretrained acoustic encoder weights loaded successfully.")

    def freeze_encoder(self, freeze_subsampling=True, freeze_pos_enc=True, freeze_conformer=True):
        if freeze_subsampling:
            for param in self.subsampling.parameters():
                param.requires_grad = False
            print("Subsampling module frozen.")
        if freeze_pos_enc:
            for param in self.positional_encoding.parameters():
                param.requires_grad = False
            print("PositionalEncoding module frozen.")
        if freeze_conformer:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("ConformerEncoder module frozen.")


# --- 微调配置 ---
class FineTuneConfig:
    # 数据和分词器路径 (与预训练时相同)
    train_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/train/wav.scp"
    train_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/train/pinyin"
    test_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/test/wav.scp"
    test_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/test/pinyin"
    
    # 模型配置 (应与预训练的声学编码器部分匹配)
    mel_input_dim = 80  # 必须与你的数据特征维度一致
    n_mels = 80         # 同上
    conformer_d_model = 256
    conformer_num_layers = 6 # 与预训练时一致
    conformer_nhead = 8      # 与预训练时一致
    conformer_ff_ratio = 4
    conformer_dropout_rate = 0.1
    conformer_conv_kernel = 31
    conformer_pos_enc_dropout_rate = 0.1
    conformer_subsampling_factor = 8 # 与预训练时一致

    # 微调特定参数
    epochs = 30
    batch_size = 64 # 根据显存调整
    learning_rate = 5e-4 # 微调时学习率通常较小
    encoder_learning_rate = 5e-4 # 如果想对encoder使用更小的学习率
    freeze_encoder_epochs = 0 # 前N个epoch冻结encoder，只训练输出层
    grad_clip = 5.0
    
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    seed = 42
    num_workers = 0 # 根据你的环境设置
    
    # 检查点路径
    pretrained_checkpoint_path = "/data/mingxu/Audio/Assignment_3/pretrain_conformer_checkpoints/pretrain_conformer_epoch_23.pt" # 你预训练好的模型
    finetune_checkpoint_dir = "./finetune_conformer_checkpoints"
    log_file = "./finetune_asr_log.txt"
    save_interval = 1

    def __init__(self, args_override=None):
        if args_override:
            for k, v in vars(args_override).items():
                if hasattr(self, k) and v is not None:
                    setattr(self, k, v)
        self.mel_input_dim = self.n_mels # 确保一致

# --- 微调训练脚本 ---
def finetune_asr(config: FineTuneConfig, rank=0):
    set_seed(config.seed)
    os.makedirs(config.finetune_checkpoint_dir, exist_ok=True)

    asr_tokenizer = Tokenizer() # 使用你自己的Tokenizer
    
    print(f"Initializing DataLoader with num_workers={config.num_workers}")
    train_dataloader = get_dataloader(
        config.train_wav_scp, config.train_pinyin_file, 
        config.batch_size, asr_tokenizer, shuffle=True,
    )
    valid_dataloader = get_dataloader(
        config.test_wav_scp, config.test_pinyin_file,
        config.batch_size, asr_tokenizer, shuffle=False,
    )

    model = ConformerCTCASRModel(
        config, 
        vocab_size=asr_tokenizer.size(), 
        blank_id=asr_tokenizer.blk_id()
    ).to(config.device)

    # 加载预训练权重
    if config.pretrained_checkpoint_path:
        model.load_pretrained_acoustic_encoder(config.pretrained_checkpoint_path, config.device)

    # 设置优化器 (可以为不同部分设置不同学习率)
    encoder_params = list(model.subsampling.parameters()) + \
                     list(model.positional_encoding.parameters()) + \
                     list(model.encoder.parameters())
    output_layer_params = model.fc_out.parameters()

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': config.encoder_learning_rate},
        {'params': output_layer_params, 'lr': config.learning_rate}
    ], betas=(0.9, 0.98), eps=1.0e-9, weight_decay=1.0e-6)


    if rank == 0:
        log_header = "epoch,step,train_loss,val_loss,lr_encoder,lr_output,time\n"
        mode = "w" if not os.path.exists(config.log_file) else "a"
        with open(config.log_file, mode, encoding="utf-8") as f:
            if mode == "a": f.write("\n--- New Fine-tuning Run ---\n")
            f.write(log_header)

    print(f"Using device: {config.device}")
    print(f"ASR Model total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Data input dimension (mel_input_dim) CONFIGURED TO: {config.mel_input_dim}")

    for epoch_idx in range(1, config.epochs + 1):
        model.train()
        if epoch_idx <= config.freeze_encoder_epochs:
            print(f"Epoch {epoch_idx}: Encoder is FROZEN. Training output layer only.")
            model.freeze_encoder() # 冻结预训练部分
            # 此时优化器只会更新fc_out的参数，因为其他部分requires_grad=False
            # 或者调整优化器只包含fc_out的参数
            current_optimizer = torch.optim.AdamW(
                model.fc_out.parameters(), 
                lr=config.learning_rate, 
                betas=(0.9, 0.98), eps=1.0e-9, weight_decay=1.0e-6
            )
        else:
            if epoch_idx == config.freeze_encoder_epochs + 1: # 解冻
                print(f"Epoch {epoch_idx}: Unfreezing encoder for full fine-tuning.")
                for param_group in optimizer.param_groups: # 恢复学习率
                    if any(p is param_group['params'][0] for p in encoder_params): # 检查是否是encoder参数组
                        param_group['lr'] = config.encoder_learning_rate
                    else:
                        param_group['lr'] = config.learning_rate

                for param in model.parameters(): # 确保所有参数都可训练
                     param.requires_grad = True
            current_optimizer = optimizer


        total_loss_epoch = 0.
        step_count = 0
        epoch_start_time = time.time()

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                            desc=f"FineTune Epoch {epoch_idx}/{config.epochs}", ncols=130, disable=(rank!=0))

        for batch_idx, batch in progress_bar:
            step_count += 1
            batch_on_device = to_device(batch, config.device)
            mel_inputs = batch_on_device['audios'] # (B, L_max, FeatureDim=80)
            mel_input_lengths = batch_on_device['audio_lens']
            text_targets = batch_on_device['texts'] # (B, T_max_text)
            text_target_lengths = batch_on_device['text_lens']

            if rank == 0 and batch_idx == 0 and epoch_idx == 1: # 首次运行时检查维度
                actual_data_feat_dim = mel_inputs.shape[2]
                print(f"DEBUG (Fine-tune): Actual data feature dimension from Dataloader (mel_inputs.shape[2]): {actual_data_feat_dim}")
                print(f"DEBUG (Fine-tune): Configured mel_input_dim for model: {config.mel_input_dim}")
                if actual_data_feat_dim != config.mel_input_dim:
                     print(f"CRITICAL WARNING (Fine-tune): Data dim ({actual_data_feat_dim}) != Config dim ({config.mel_input_dim})")


            current_optimizer.zero_grad()
            
            try:
                _, loss, _ = model(mel_inputs, mel_input_lengths, text_targets, text_target_lengths)
            except RuntimeError as e:
                print(f"FATAL RUNTIME ERROR during model forward at epoch {epoch_idx}, batch {batch_idx}: {e}")
                print(f"  Input mel_inputs shape to model: {mel_inputs.shape}")
                print(f"  Configured mel_input_dim for Subsampling: {config.mel_input_dim}")
                print("Ensure your custom modules are correctly designed for this input dimension.")
                raise

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf CTC loss at epoch {epoch_idx}, batch {batch_idx}. Skipping batch.")
                current_optimizer.zero_grad() # 清除可能存在的NaN梯度
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            current_optimizer.step()
            
            total_loss_epoch += loss.item()
            
            lr_encoder_disp = current_optimizer.param_groups[0]['lr']
            lr_output_disp = current_optimizer.param_groups[1]['lr'] if len(current_optimizer.param_groups) > 1 else current_optimizer.param_groups[0]['lr']

            progress_bar.set_postfix(
                ctc_loss=f"{loss.item():.4f}",
                avg_loss=f"{(total_loss_epoch / step_count):.4f}",
                lr_enc=f"{lr_encoder_disp:.2e}",
                lr_out=f"{lr_output_disp:.2e}"
            )
        
        progress_bar.close()
        avg_train_loss = total_loss_epoch / step_count if step_count > 0 else 0

        # Validation loop
        model.eval()
        total_val_loss = 0.
        val_step_count = 0
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(tqdm(valid_dataloader, desc=f"Validating Epoch {epoch_idx}", ncols=130, disable=(rank!=0))):
                val_step_count +=1
                val_batch_on_device = to_device(val_batch, config.device)
                val_mel_inputs = val_batch_on_device['audios']
                val_mel_input_lengths = val_batch_on_device['audio_lens']
                val_text_targets = val_batch_on_device['texts']
                val_text_target_lengths = val_batch_on_device['text_lens']

                _, val_loss, _ = model(val_mel_inputs, val_mel_input_lengths, val_text_targets, val_text_target_lengths)
                if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                    total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / val_step_count if val_step_count > 0 else 0
        epoch_duration = time.time() - epoch_start_time
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        lr_e = current_optimizer.param_groups[0]['lr']
        lr_o = current_optimizer.param_groups[1]['lr'] if len(current_optimizer.param_groups) > 1 else lr_e
        
        log_msg = (f"\n[{current_time_str}] FineTune Epoch {epoch_idx} Done | Time: {epoch_duration:.2f}s\n"
                   f"  Train CTC Loss: {avg_train_loss:.4f}\n"
                   f"  Valid CTC Loss: {avg_val_loss:.4f}\n"
                   f"  LR Encoder: {lr_e:.2e}, LR Output: {lr_o:.2e}\n")
        print(log_msg)
        if rank == 0:
            with open(config.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch_idx},{step_count},{avg_train_loss:.4f},{avg_val_loss:.4f},"
                        f"{lr_e:.2e},{lr_o:.2e},{current_time_str}\n")
            
        if rank == 0 and (epoch_idx % config.save_interval == 0 or epoch_idx == config.epochs):
            ckpt_path = os.path.join(config.finetune_checkpoint_dir, f"finetune_asr_epoch_{epoch_idx}.pt")
            torch.save({
                'epoch': epoch_idx,
                'config': vars(config), # 保存微调配置
                'model_state_dict': model.state_dict(), # 保存整个ASR模型的状态
                'optimizer_state_dict': current_optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved fine-tuned ASR model checkpoint to: {ckpt_path}")

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Conformer ASR with Pre-trained Encoder")
    parser.add_argument('--epochs', type=int, help="Number of fine-tuning epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size for fine-tuning")
    parser.add_argument('--lr', type=float, dest='learning_rate', help="Learning rate for output layer")
    parser.add_argument('--lr_encoder', type=float, dest='encoder_learning_rate', help="Learning rate for pre-trained encoder")
    parser.add_argument('--pretrained_ckpt', type=str, dest='pretrained_checkpoint_path', help="Path to the pre-trained encoder checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, dest='finetune_checkpoint_dir', help="Directory to save fine-tuned checkpoints")
    parser.add_argument('--log_file', type=str, help="Log file name for fine-tuning")
    parser.add_argument('--n_mels', type=int, help="Number of Mel features (should be 80 to match data)")
    parser.add_argument('--device', type=str, help="e.g., cuda:0 or cpu")
    parser.add_argument('--num_workers', type=int, help="DataLoader num_workers")
    parser.add_argument('--freeze_encoder_epochs', type=int, help="Number of initial epochs to freeze the encoder")


    args = parser.parse_args()
    current_config = FineTuneConfig(args_override=args) 

    print("--- Starting ASR Fine-tuning ---")
    print("--- Using Fine-tune Configuration: ---")
    for key, value in vars(current_config).items():
        if not key.startswith('__'): print(f"  {key}: {value}")
    print("--------------------")

    finetune_asr(current_config)

if __name__ == '__main__':
    main()