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
from torch.utils.data import Dataset, DataLoader # 从 torch.utils.data 导入
import torchaudio
from torchaudio.transforms import MelSpectrogram
import numpy as np
from transformers import BertModel, BertTokenizerFast
from tqdm import tqdm
import matplotlib.pyplot as plt
import time # 确保导入 time

def set_seed(seed_value):
    random.seed(seed_value); np.random.seed(seed_value); torch.manual_seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

# --- 用户本地模块的导入 ---
try:
    # 脚本会尝试从这些路径导入你本地的模块
    from data.dataloader import get_dataloader
    from tokenizer.tokenizer import Tokenizer
    from utils.utils import to_device
    from model.model import ConformerEncoder, PositionalEncoding, Subsampling
    print("Successfully imported user local modules (dataloader, tokenizer, utils, model.model).")
except ImportError as e:
    print(f"Could not import user-defined modules: {e}")
    print("Ensure the script is run from the correct project directory or sys.path is configured.")
    print("Using placeholder implementations for missing modules because local import failed.")
    # 下面是导入失败时的占位符实现
    class Tokenizer:
        def __init__(self): self.word2idx = {'<blk>': 0, '<unk>':1, 'a':2, 'b':3}; self.idx2word = {v:k for k,v in self.word2idx.items()}
        def size(self): return len(self.word2idx)
        def blk_id(self): return self.word2idx['<blk>']
        def encode(self, text_list): return [[self.word2idx.get(c, 1) for c in text] for text in text_list]
        def decode(self, id_list): return [str(self.idx2word.get(int(i), '<unk>')) for i in id_list if isinstance(i, (int, float))]
        def decode_to_string(self, id_list): return "".join(self.decode(id_list))

    def get_dataloader(wav_scp, pinyin_file, batch_size, tokenizer, shuffle=True, num_workers_override=0):
        print(f"WARNING: Using PLACEHOLDER get_dataloader. Feature dim will be 80. Output shape (B, L, D).")
        class DummyDataset(Dataset):
            def __init__(self, feature_dim=80):
                self.data = []
                self.feature_dim = feature_dim # 特征维度在后
                for _ in range(100):
                    audio_len = random.randint(100,300); text_len = random.randint(5,15)
                    # Dummy data: (Time, FeatureDim) to match user's extract_audio_features
                    self.data.append({
                        'audios': torch.randn(audio_len, self.feature_dim), 
                        'audio_lens': torch.tensor([audio_len]), 
                        'texts': torch.randint(0, tokenizer.size(), (text_len,)), 
                        'text_lens': torch.tensor([text_len])
                    })
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx]
        
        def placeholder_collate_fn(batch): # 模拟用户 collate_with_PAD 的输出形状
            audios = [item['audios'] for item in batch] # list of (L, D)
            texts = [item['texts'] for item in batch]
            audio_lens = torch.tensor([s.size(0) for s in audios], dtype=torch.long)
            text_lens = torch.tensor([len(s) for s in texts], dtype=torch.long)
            
            # Pad audios to (B, L_max, D)
            audios_padded = nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0)
            
            # Pad texts
            texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.blk_id())
            
            return {'audios': audios_padded, 'audio_lens': audio_lens, 
                    'texts': texts_padded, 'text_lens': text_lens}

        return DataLoader(DummyDataset(feature_dim=80), batch_size=batch_size, 
                          collate_fn=placeholder_collate_fn, num_workers=num_workers_override)

    def to_device(batch, device):
        if isinstance(batch, dict): return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch.to(device) if isinstance(batch, torch.Tensor) else batch

    class Subsampling(nn.Module): 
        def __init__(self, in_dim, out_dim, subsampling_type=8): # in_dim is feature_dim (80)
            super().__init__(); print("WARNING: Using placeholder Subsampling")
            # Placeholder Subsampling expects (B, Time, FeatureDim)
            # This linear layer will project from in_dim (80) to out_dim (model_dim)
            # after subsampling in time.
            # A more realistic one would use Conv layers.
            self.conv_subsample = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=2, padding=1) # For 8x
            )
            self.subsampling_rate = subsampling_type # Should be 8 for 3 conv layers with stride 2

        def forward(self, x, x_lens): # x is (B, Time, FeatureDim)
            x = x.permute(0, 2, 1) # (B, FeatureDim, Time) for Conv1d
            x = self.conv_subsample(x)
            x = x.permute(0, 2, 1) # (B, SubsampledTime, ModelDim)
            
            # Update lengths based on conv layers
            for _ in range(3): # 3 conv layers with stride 2
                 x_lens = (x_lens - 1) // 2 + 1

            max_len_out = x_lens.max().item() if x_lens.numel() > 0 else 0
            x = x[:, :max_len_out, :] if max_len_out > 0 else x[:, :0, :]
            return x, x_lens

    class PositionalEncoding(nn.Module): 
        def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
            super().__init__(); self.d_model = d_model; self.xscale = math.sqrt(self.d_model)
            self.dropout = nn.Dropout(p=dropout_rate); self.pe = None; self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        def extend_pe(self, x: torch.Tensor):
            if self.pe is not None and self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device: self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
            pe = torch.zeros(x.size(1), self.d_model); position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0).to(device=x.device, dtype=x.dtype)
        def forward(self, x: torch.Tensor, dummy_offset=0): # x is (B, Time, Dim)
            self.extend_pe(x); x = x * self.xscale + self.pe[:, :x.size(1)]; return self.dropout(x), self.dropout(self.pe[:, :x.size(1)])

    class ConformerEncoder(nn.Module):
        def __init__(self, num_layers, d_model, nhead, ff_ratio=4, dropout=0.1, conv_kernel=31, **kwargs):
            super().__init__(); print("WARNING: Using placeholder ConformerEncoder"); self.fc = nn.Linear(d_model, d_model)
        def forward(self, x, mask=None): return self.fc(x)

warnings.simplefilter(action='ignore', category=FutureWarning)
torch.backends.cudnn.benchmark = False
class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x * torch.sigmoid(x)
class SimpleMelDecoder(nn.Module):
    def __init__(self, input_dim: int, output_mel_dim: int, upsample_factor: int = 8, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__(); self.input_dim = input_dim; self.output_mel_dim = output_mel_dim; layers = []; current_dim = input_dim
        num_upsample_layers = 0
        if upsample_factor > 1:
            num_upsample_layers = int(round(math.log2(upsample_factor))) if upsample_factor > 0 else 0
            for i in range(num_upsample_layers):
                out_c = hidden_dim; layers.append(nn.ConvTranspose1d(current_dim, out_c, kernel_size=4, stride=2, padding=1))
                layers.append(nn.InstanceNorm1d(out_c)); layers.append(nn.LeakyReLU(0.2)); current_dim = out_c
        for _ in range(num_layers):
            layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=7, padding=3))
            layers.append(nn.InstanceNorm1d(hidden_dim)); layers.append(nn.LeakyReLU(0.2)); current_dim = hidden_dim
        layers.append(nn.Conv1d(current_dim, output_mel_dim, kernel_size=7, padding=3)); self.decoder = nn.Sequential(*layers)
        approx_total_upsample = 2**num_upsample_layers if num_upsample_layers > 0 else 1
        print(f"SimpleMelDecoder initialized: input_dim={input_dim}, output_mel_dim={output_mel_dim}, upsample_factor={upsample_factor} (approx {approx_total_upsample}x)")
    def forward(self, conformer_sequence_output): # Input (B, T_sub, D_model)
        x = conformer_sequence_output.permute(0, 2, 1) # (B, D_model, T_sub)
        reconstructed_mel = self.decoder(x) # Output (B, output_mel_dim, T_reconstructed)
        return reconstructed_mel
class CLReconConformerASRPretrainModel(nn.Module):
    def __init__(self, config, tokenizer_vocab_size, tokenizer_blank_id):
        super().__init__(); self.config = config
        # Subsampling expects input (B, Time, FeatureDim=80)
        self.subsampling = Subsampling(config.mel_input_dim, config.conformer_d_model, subsampling_type=config.conformer_subsampling_factor)
        self.positional_encoding = PositionalEncoding(config.conformer_d_model, config.conformer_pos_enc_dropout_rate)
        self.acoustic_encoder = ConformerEncoder(num_layers=config.conformer_num_layers, d_model=config.conformer_d_model, nhead=config.conformer_nhead, ff_ratio=config.conformer_ff_ratio, dropout=config.conformer_dropout_rate, conv_kernel=config.conformer_conv_kernel)
        self.speech_cl_projection_head = nn.Linear(config.conformer_d_model, config.cl_embedding_dim)
        print(f"Loading Text Encoder: {config.text_encoder_model_name}")
        self.text_encoder_bert = BertModel.from_pretrained(config.text_encoder_model_name); [p.requires_grad_(False) for p in self.text_encoder_bert.parameters()]; self.text_encoder_bert.eval()
        self.text_cl_projection_head = nn.Linear(config.text_encoder_output_dim, config.cl_embedding_dim)
        if config.lambda_mel_reconstruction > 0: self.mel_decoder = SimpleMelDecoder(config.conformer_d_model, config.mel_input_dim, config.decoder_upsample_factor, config.decoder_hidden_dim, config.decoder_num_conv_layers)
        else: self.mel_decoder = None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.cl_temperature))
    
    def encode_speech(self, mel_inputs_from_dataloader, mel_input_lengths):
        # mel_inputs_from_dataloader shape: (B, L_max, FeatureDim=80)
        # self.subsampling (user's or placeholder) expects (B, Time, FeatureDim)
        subsampled_feat, subsampled_lens = self.subsampling(mel_inputs_from_dataloader, mel_input_lengths)
        # subsampled_feat shape: (B, L_sub, D_model)
        encoded_seq, _ = self.positional_encoding(subsampled_feat)
        max_sub_len = encoded_seq.size(1)
        conformer_mask = torch.arange(max_sub_len, device=encoded_seq.device)[None, :] >= subsampled_lens[:, None]
        conformer_out_seq = self.acoustic_encoder(encoded_seq, conformer_mask)
        valid_frame_mask = ~conformer_mask
        masked_sequence = conformer_out_seq * valid_frame_mask.unsqueeze(-1)
        pooled_speech_features = masked_sequence.sum(dim=1) / subsampled_lens.unsqueeze(1).clamp(min=1)
        speech_cl_embedding = F.normalize(self.speech_cl_projection_head(pooled_speech_features), dim=-1)
        return speech_cl_embedding, conformer_out_seq, subsampled_lens

    def encode_text(self, text_input_ids, text_attention_masks):
        with torch.no_grad(): bert_outputs = self.text_encoder_bert(input_ids=text_input_ids, attention_mask=text_attention_masks)
        last_hidden = bert_outputs.last_hidden_state; attention_mask_expanded = text_attention_masks.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * attention_mask_expanded, 1); sum_mask = attention_mask_expanded.sum(1).clamp(min=1e-9); text_pooled_features = sum_embeddings / sum_mask
        text_cl_embedding = F.normalize(self.text_cl_projection_head(text_pooled_features), dim=-1); return text_cl_embedding

    def forward(self, mel_inputs_from_dataloader, mel_input_lengths, text_input_ids, text_attention_masks, target_mels_for_recon):
        # mel_inputs_from_dataloader shape: (B, L_max, FeatureDim=80)
        # target_mels_for_recon shape: (B, L_max, FeatureDim=80)
        speech_cl_embedding, conformer_seq_for_recon, conformer_out_lengths = self.encode_speech(mel_inputs_from_dataloader, mel_input_lengths)
        text_cl_embedding = self.encode_text(text_input_ids, text_attention_masks); logit_scale = self.logit_scale.exp()
        logits_per_speech = logit_scale * speech_cl_embedding @ text_cl_embedding.t(); logits_per_text = logits_per_speech.t()
        batch_size_cl = speech_cl_embedding.shape[0]; cl_labels = torch.arange(batch_size_cl, device=speech_cl_embedding.device)
        loss_cl = (F.cross_entropy(logits_per_speech, cl_labels) + F.cross_entropy(logits_per_text, cl_labels)) / 2.0
        with torch.no_grad():
            preds_speech = torch.argmax(logits_per_speech, dim=1); acc_speech = (preds_speech == cl_labels).float().mean()
            preds_text = torch.argmax(logits_per_text, dim=1); acc_text = (preds_text == cl_labels).float().mean()
            accuracy_cl = (acc_speech + acc_text) / 2.0
        loss_recon = torch.tensor(0.0, device=loss_cl.device); reconstructed_mel_for_plot = None
        
        if self.mel_decoder is not None and self.config.lambda_mel_reconstruction > 0:
            # conformer_seq_for_recon shape: (B, L_sub, D_model)
            # reconstructed_mel_seq shape: (B, output_mel_dim=80, T_reconstructed)
            reconstructed_mel_seq = self.mel_decoder(conformer_seq_for_recon)
            
            # target_mels_for_recon shape: (B, L_max_data, FeatureDim_data=80)
            # Permute target to match reconstructed_mel_seq's layout (B, FeatureDim, Time)
            target_mels_for_loss_permuted = target_mels_for_recon.permute(0, 2, 1)
            # target_mels_for_loss_permuted shape: (B, FeatureDim_data=80, L_max_data)

            reconstructed_feature_dim = reconstructed_mel_seq.shape[1] # Should be 80
            actual_target_feature_dim = target_mels_for_loss_permuted.shape[1] # Should be 80

            if reconstructed_feature_dim != actual_target_feature_dim:
                print(f"ERROR: Feature dimension mismatch for reconstruction. Reconstructed: {reconstructed_feature_dim}, Target: {actual_target_feature_dim}.")
                loss_recon = torch.tensor(0.0, device=loss_cl.device) 
            else:
                # Compare time dimensions
                min_len_time = min(reconstructed_mel_seq.size(2), target_mels_for_loss_permuted.size(2))
            
                # mel_input_lengths are original time lengths (before any subsampling/padding by dataloader for L_max)
                # We need a mask for the common time dimension min_len_time
                # This mask should be based on the true lengths of the sequences in the batch, up to min_len_time
                time_mask_for_loss = torch.arange(min_len_time, device=target_mels_for_loss_permuted.device)[None, :] < mel_input_lengths[:, None]
                
                # Expand mask to (B, FeatureDim=80, min_len_time)
                final_recon_mask = time_mask_for_loss.unsqueeze(1).expand(-1, actual_target_feature_dim, -1)
                
                recon_mel_to_loss = reconstructed_mel_seq[:, :, :min_len_time]
                target_mel_to_loss_final = target_mels_for_loss_permuted[:, :, :min_len_time]

                masked_reconstructed = recon_mel_to_loss[final_recon_mask]
                masked_target = target_mel_to_loss_final[final_recon_mask]
                
                if masked_target.numel() > 0: loss_recon = F.l1_loss(masked_reconstructed, masked_target)
                else: loss_recon = torch.tensor(0.0, device=loss_cl.device)

            global current_epoch_for_plot, current_batch_idx_for_plot
            if current_epoch_for_plot % self.config.save_sample_plot_interval == 0 and current_batch_idx_for_plot == 0:
                 if reconstructed_mel_seq.numel() > 0 : reconstructed_mel_for_plot = reconstructed_mel_seq[0].cpu().detach() # Shape (80, T_reconstructed)
        
        total_loss = loss_cl + self.config.lambda_mel_reconstruction * loss_recon
        return total_loss, loss_cl, loss_recon, accuracy_cl, reconstructed_mel_for_plot

# --- 配置类 (适配 ASR 预训练) ---
class PretrainConfig:
    train_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/train/wav.scp"
    train_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/train/pinyin"
    test_wav_scp = "/data/mingxu/Audio/Assignment_3/dataset/split/test/wav.scp"
    test_pinyin_file = "/data/mingxu/Audio/Assignment_3/dataset/split/test/pinyin"
    mel_input_dim = 80 # Crucial: Should match dataloader output
    n_mels = 80        # Crucial: Should match dataloader output
    conformer_d_model = 256; conformer_num_layers = 6; conformer_nhead = 8
    conformer_ff_ratio = 4; conformer_dropout_rate = 0.1; conformer_conv_kernel = 31
    conformer_pos_enc_dropout_rate = 0.1; conformer_subsampling_factor = 8
    text_encoder_model_name = "bert-base-chinese"; text_encoder_output_dim = 768
    cl_embedding_dim = 256; cl_temperature = 0.07; decoder_hidden_dim = 256
    decoder_num_conv_layers = 3; decoder_upsample_factor = 8
    lambda_mel_reconstruction = 1.0; epochs = 30; batch_size = 64; learning_rate = 1e-4
    accum_steps = 1; grad_clip = 5.0; device = "cuda:2" if torch.cuda.is_available() else "cpu"
    seed = 42; num_workers = 0; log_file = "./pretrain_cl_recon_log.txt"
    checkpoint_dir = "./pretrain_conformer_checkpoints"; save_sample_plot_interval = 1; save_interval = 1

    def __init__(self, args_override=None):
        if args_override:
            for k, v in vars(args_override).items():
                if hasattr(self, k) and v is not None: setattr(self, k, v)
        self.mel_input_dim = self.n_mels
        self.decoder_upsample_factor = self.conformer_subsampling_factor

# --- 训练脚本 ---
current_epoch_for_plot = 0; current_batch_idx_for_plot = 0
def pretrain_conformer_asr(config: PretrainConfig, rank=0):
    global current_epoch_for_plot, current_batch_idx_for_plot; set_seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True); plot_output_dir = os.path.join(config.checkpoint_dir, "plots"); os.makedirs(plot_output_dir, exist_ok=True)
    asr_tokenizer = Tokenizer(); bert_tokenizer = BertTokenizerFast.from_pretrained(config.text_encoder_model_name)
    print(f"Initializing DataLoader with num_workers={config.num_workers}")
    # Ensure num_workers_override is passed if get_dataloader expects it
    train_dataloader = get_dataloader(config.train_wav_scp, config.train_pinyin_file, config.batch_size, asr_tokenizer, shuffle=True)
    valid_dataloader = get_dataloader(config.test_wav_scp, config.test_pinyin_file, config.batch_size, asr_tokenizer, shuffle=False)
    
    model = CLReconConformerASRPretrainModel(config, asr_tokenizer.size(), asr_tokenizer.blk_id()).to(config.device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, betas=(0.9, 0.98), eps=1.0e-9, weight_decay=1.0e-6)
    if rank == 0:
        log_header = "epoch,step,total_loss,cl_loss,recon_loss,cl_acc,val_total_loss,val_cl_loss,val_recon_loss,val_cl_acc,lr,time\n"
        mode = "w" if not os.path.exists(config.log_file) else "a"
        with open(config.log_file, mode, encoding="utf-8") as f:
            if mode == "a": f.write("\n--- New Training Run ---\n")
            f.write(log_header)
    print(f"Using device: {config.device}"); print(f"Model total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"); print(f"Data input dimension (mel_input_dim) CONFIGURED TO: {config.mel_input_dim}")
    
    for epoch_idx in range(1, config.epochs + 1):
        current_epoch_for_plot = epoch_idx; model.train(); model.text_encoder_bert.eval()
        total_loss_epoch, cl_loss_epoch, recon_loss_epoch, cl_acc_epoch = 0., 0., 0., 0.; step_count = 0; epoch_start_time = time.time()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Pretrain Epoch {epoch_idx}/{config.epochs}", ncols=130)
        
        for batch_idx, batch in progress_bar:
            current_batch_idx_for_plot = batch_idx; step_count += 1
            batch_on_device = to_device(batch, config.device); 
            mel_inputs = batch_on_device['audios'] # Expected shape (B, L_max, FeatureDim=80)
            mel_input_lengths = batch_on_device['audio_lens']
            pinyin_ids_asr = batch_on_device['texts']; pinyin_lens_asr = batch_on_device['text_lens']
            
            if rank == 0 and batch_idx == 0 and epoch_idx == 1:
                actual_data_feat_dim_loader = mel_inputs.shape[2] # Correct index for feature dim
                print(f"DEBUG: Actual data feature dimension from Dataloader (mel_inputs.shape[2]): {actual_data_feat_dim_loader}")
                print(f"DEBUG: Configured mel_input_dim for model: {config.mel_input_dim}")
                if actual_data_feat_dim_loader != config.mel_input_dim:
                    print(f"CRITICAL WARNING: ACTUAL data feature dimension ({actual_data_feat_dim_loader}) "
                          f"from Dataloader does NOT match CONFIGURED mel_input_dim ({config.mel_input_dim}).")
                    print("This WILL cause errors. Ensure your Dataloader ('data/dataloader.py -> extract_audio_features') "
                          "produces features of the configured dimension (n_mels), "
                          "or adjust PretrainConfig.n_mels accordingly.")

            pinyin_strings_batch = []
            for b_idx in range(pinyin_ids_asr.size(0)):
                ids = pinyin_ids_asr[b_idx, :pinyin_lens_asr[b_idx]].cpu().tolist()
                if hasattr(asr_tokenizer, 'decode_to_string') and callable(getattr(asr_tokenizer, 'decode_to_string')): pinyin_str = asr_tokenizer.decode_to_string(ids)
                else: 
                    pinyin_str_list = asr_tokenizer.decode(ids)
                    if isinstance(pinyin_str_list, list): pinyin_str = "".join(pinyin_str_list)
                    elif isinstance(pinyin_str_list, str): pinyin_str = pinyin_str_list
                    else: pinyin_str = ""
                if not isinstance(pinyin_str, str): pinyin_str = ""
                pinyin_strings_batch.append(pinyin_str)

            if not all(isinstance(s, str) for s in pinyin_strings_batch): print(f"ERROR: pinyin_strings_batch contains non-strings. Skipping batch."); continue
            tokenized_bert_texts = bert_tokenizer(pinyin_strings_batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            text_input_ids_bert = tokenized_bert_texts.input_ids.to(config.device); text_attention_masks_bert = tokenized_bert_texts.attention_mask.to(config.device)
            optimizer.zero_grad()
            try:
                total_loss, cl_loss, recon_loss, cl_accuracy, reconstructed_mel_plot = model(mel_inputs, mel_input_lengths, text_input_ids_bert, text_attention_masks_bert, mel_inputs)
            except RuntimeError as e:
                print(f"FATAL RUNTIME ERROR during model forward at epoch {epoch_idx}, batch {batch_idx}: {e}")
                print(f"  Input mel_inputs shape to model: {mel_inputs.shape}")
                print(f"  Configured mel_input_dim for Subsampling: {config.mel_input_dim}")
                print("Ensure your custom modules in 'model/model.py' and 'model/subsampling.py' are correctly designed for this input dimension.")
                raise
            if torch.isnan(total_loss) or torch.isinf(total_loss): print(f"WARNING: NaN/Inf total loss at epoch {epoch_idx}, batch {batch_idx}. Skipping batch update."); optimizer.zero_grad(); continue
            loss_to_backward = total_loss; loss_to_backward.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip); optimizer.step()
            total_loss_epoch += total_loss.item(); cl_loss_epoch += cl_loss.item(); recon_loss_epoch += recon_loss.item(); cl_acc_epoch += cl_accuracy.item()
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix(total_loss=f"{total_loss.item():.4f}", cl_loss=f"{cl_loss.item():.4f}", recon_loss=f"{recon_loss.item():.4f}", cl_acc=f"{cl_accuracy.item():.3f}", lr=f"{current_lr:.2e}")
            
            if rank == 0 and epoch_idx % config.save_sample_plot_interval == 0 and batch_idx == 0 and reconstructed_mel_plot is not None:
                if reconstructed_mel_plot.numel() > 0 and mel_inputs.numel() > 0: # mel_inputs shape (B, L, D=80)
                    plt.figure(figsize=(12, 6)); 
                    plt.subplot(1, 2, 1); 
                    # Target mel_inputs[0] is (L, 80), transpose to (80, L) for imshow
                    plt.imshow(mel_inputs[0].cpu().transpose(0, 1).numpy(), aspect='auto', origin='lower', interpolation='none')
                    plt.title(f"Target Mel (Train Sample) Epoch {epoch_idx}\nShape: {mel_inputs[0].transpose(0,1).shape} (Freq, Time)")
                    plt.xlabel("Time Frames"); plt.ylabel("Mel Bins")
                    
                    plt.subplot(1, 2, 2); 
                    # reconstructed_mel_plot is (80, T_reconstructed)
                    min_len_vis_time = min(reconstructed_mel_plot.size(1), mel_inputs[0].size(0)) # Compare time dimensions
                    plt.imshow(reconstructed_mel_plot[:, :min_len_vis_time].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
                    plt.title(f"Reconstructed Mel (Train Sample) Epoch {epoch_idx}\nShape: {reconstructed_mel_plot.shape} (Freq, Time)")
                    plt.xlabel("Time Frames"); plt.ylabel("Mel Bins")
                    plt.tight_layout(); plt.savefig(os.path.join(plot_output_dir, f"mel_recon_train_epoch_{epoch_idx}.png")); plt.close()
        
        progress_bar.close()
        avg_total_loss = total_loss_epoch / step_count if step_count > 0 else 0; avg_cl_loss = cl_loss_epoch / step_count if step_count > 0 else 0; avg_recon_loss = recon_loss_epoch / step_count if step_count > 0 else 0; avg_cl_acc = cl_acc_epoch / step_count if step_count > 0 else 0
        model.eval(); val_total_loss_epoch, val_cl_loss_epoch, val_recon_loss_epoch, val_cl_acc_epoch = 0., 0., 0., 0.; val_step_count = 0
        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(tqdm(valid_dataloader, desc=f"Validating Epoch {epoch_idx}", ncols=130, disable=(rank!=0))):
                val_step_count += 1; val_batch = to_device(val_batch, config.device); val_mel_inputs = val_batch['audios']; val_mel_input_lengths = val_batch['audio_lens']; val_pinyin_ids_asr = val_batch['texts']; val_pinyin_lens_asr = val_batch['text_lens']
                val_pinyin_strings_batch = []
                for b_idx in range(val_pinyin_ids_asr.size(0)):
                    ids = val_pinyin_ids_asr[b_idx, :val_pinyin_lens_asr[b_idx]].cpu().tolist()
                    if hasattr(asr_tokenizer, 'decode_to_string') and callable(getattr(asr_tokenizer, 'decode_to_string')): pinyin_str = asr_tokenizer.decode_to_string(ids)
                    else: pinyin_str_list = asr_tokenizer.decode(ids); pinyin_str = "".join(pinyin_str_list) if isinstance(pinyin_str_list, list) else pinyin_str_list
                    if not isinstance(pinyin_str, str): pinyin_str = ""
                    val_pinyin_strings_batch.append(pinyin_str)
                val_tokenized_bert_texts = bert_tokenizer(val_pinyin_strings_batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                val_text_input_ids_bert = val_tokenized_bert_texts.input_ids.to(config.device); val_text_attention_masks_bert = val_tokenized_bert_texts.attention_mask.to(config.device)
                val_total_loss, val_cl_loss, val_recon_loss, val_cl_accuracy, val_recon_plot_val = model(val_mel_inputs, val_mel_input_lengths, val_text_input_ids_bert, val_text_attention_masks_bert, val_mel_inputs)
                if not (torch.isnan(val_total_loss) or torch.isinf(val_total_loss)):
                    val_total_loss_epoch += val_total_loss.item(); val_cl_loss_epoch += val_cl_loss.item(); val_recon_loss_epoch += val_recon_loss.item(); val_cl_acc_epoch += val_cl_accuracy.item()
                
                if rank == 0 and epoch_idx % config.save_sample_plot_interval == 0 and val_batch_idx == 0 and val_recon_plot_val is not None:
                     if val_recon_plot_val.numel() > 0 and val_mel_inputs.numel() > 0: # val_mel_inputs shape (B, L, D=80)
                        plt.figure(figsize=(12, 6)); 
                        plt.subplot(1, 2, 1); 
                        plt.imshow(val_mel_inputs[0].cpu().transpose(0,1).numpy(), aspect='auto', origin='lower', interpolation='none')
                        plt.title(f"Target Mel (Valid Sample) Epoch {epoch_idx}\nShape: {val_mel_inputs[0].transpose(0,1).shape} (Freq, Time)")
                        plt.xlabel("Time Frames"); plt.ylabel("Mel Bins")
                        plt.subplot(1, 2, 2); 
                        min_len_vis_val_time = min(val_recon_plot_val.size(1), val_mel_inputs[0].size(0))
                        plt.imshow(val_recon_plot_val[:, :min_len_vis_val_time].cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
                        plt.title(f"Reconstructed Mel (Valid Sample) Epoch {epoch_idx}\nShape: {val_recon_plot_val.shape} (Freq, Time)")
                        plt.xlabel("Time Frames"); plt.ylabel("Mel Bins")
                        plt.tight_layout(); plt.savefig(os.path.join(plot_output_dir, f"mel_recon_valid_epoch_{epoch_idx}.png")); plt.close()
        
        avg_val_total_loss = val_total_loss_epoch / val_step_count if val_step_count > 0 else 0; avg_val_cl_loss = val_cl_loss_epoch / val_step_count if val_step_count > 0 else 0; avg_val_recon_loss = val_recon_loss_epoch / val_step_count if val_step_count > 0 else 0; avg_val_cl_acc = val_cl_acc_epoch / val_step_count if val_step_count > 0 else 0
        epoch_duration = time.time() - epoch_start_time; current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_msg = (f"\n[{current_time_str}] Epoch {epoch_idx} Done | Time: {epoch_duration:.2f}s\n"
                   f"  Train -> Total Loss: {avg_total_loss:.4f}, CL Loss: {avg_cl_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, CL Acc: {avg_cl_acc:.3f}\n"
                   f"  Valid -> Total Loss: {avg_val_total_loss:.4f}, CL Loss: {avg_val_cl_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}, CL Acc: {avg_val_cl_acc:.3f}\n"
                   f"  LR: {current_lr:.2e}\n"); print(log_msg)
        if rank == 0:
            with open(config.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{epoch_idx},{step_count},{avg_total_loss:.4f},{avg_cl_loss:.4f},{avg_recon_loss:.4f},{avg_cl_acc:.3f},"
                        f"{avg_val_total_loss:.4f},{avg_val_cl_loss:.4f},{avg_val_recon_loss:.4f},{avg_val_cl_acc:.3f},"
                        f"{current_lr},{current_time_str}\n")
        if rank == 0 and (epoch_idx % config.save_interval == 0 or epoch_idx == config.epochs):
            ckpt_path = os.path.join(config.checkpoint_dir, f"pretrain_conformer_epoch_{epoch_idx}.pt")
            save_dict = { 'epoch': epoch_idx, 'config': vars(config), 'optimizer_state_dict': optimizer.state_dict(), 'logit_scale': model.logit_scale.data }
            if hasattr(model, 'subsampling'): save_dict['subsampling_state_dict'] = model.subsampling.state_dict()
            if hasattr(model, 'positional_encoding'): save_dict['positional_encoding_state_dict'] = model.positional_encoding.state_dict()
            if hasattr(model, 'acoustic_encoder'): save_dict['acoustic_encoder_state_dict'] = model.acoustic_encoder.state_dict()
            if hasattr(model, 'speech_cl_projection_head'): save_dict['speech_cl_projection_head_state_dict'] = model.speech_cl_projection_head.state_dict()
            if hasattr(model, 'text_cl_projection_head'): save_dict['text_cl_projection_head_state_dict'] = model.text_cl_projection_head.state_dict()
            if hasattr(model, 'mel_decoder') and model.mel_decoder is not None: save_dict['mel_decoder_state_dict'] = model.mel_decoder.state_dict()
            torch.save(save_dict, ckpt_path); print(f"Saved pre-trained model checkpoint to: {ckpt_path}")

# --- 主函数 ---
def main():
    parser = argparse.ArgumentParser(description="Stage 1: CL + MelRecon Pre-training for Conformer ASR")
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--lr', type=float, dest='learning_rate', help="Learning rate")
    parser.add_argument('--lambda_mel_recon', type=float, dest='lambda_mel_reconstruction', help="Weight for mel reconstruction loss")
    parser.add_argument('--checkpoint_dir', type=str, help="Directory to save checkpoints")
    parser.add_argument('--log_file', type=str, help="Log file name")
    parser.add_argument('--conformer_d_model', type=int)
    parser.add_argument('--conformer_num_layers', type=int)
    parser.add_argument('--conformer_nhead', type=int)
    parser.add_argument('--n_mels', type=int, help="Number of Mel features. Default in PretrainConfig is 80.") # CLI arg still exists
    parser.add_argument('--device', type=str, help="e.g., cuda:0 or cpu")
    parser.add_argument('--num_workers', type=int, default=None, help="DataLoader num_workers. Default uses PretrainConfig value.")
    args = parser.parse_args()
    current_config = PretrainConfig(args_override=args) 
    print("--- Starting Pre-training Conformer ASR (Stage 1) ---")
    print("--- Using Configuration: ---")
    for key, value in vars(current_config).items():
        if not key.startswith('__'): print(f"  {key}: {value}")
    print("--------------------")
    pretrain_conformer_asr(current_config)

if __name__ == '__main__':
    main()