import os
import torch
from tqdm import tqdm
from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.model import CTCModel, Conformer_CTCModel
from utils.utils import to_device

def greedy_search(ctc_probs: torch.tensor, encoder_out_lens: torch.tensor, blank_id: int):
    """
    Args:
        ctc_probs: [B, T, V] log_probs or probs, 通常用exp或softmax得到
        encoder_out_lens: [B]
        blank_id: int, CTC blank token id
    Returns:
        hyps: List[List[int]]，每句话的预测id序列
    """
    batch_size, maxlen = ctc_probs.size()[:2]
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)
    topk_index = topk_index.view(batch_size, maxlen)
    encoder_out_lens = encoder_out_lens.view(-1).tolist()

    hyps = []

    for i in range(len(encoder_out_lens)):
        ids = topk_index[i, :encoder_out_lens[i]].tolist()
        hyp = []
        prev = None
        for idx in ids:
            # 跳过blank和重复
            if idx != blank_id and idx != prev:
                hyp.append(idx)
            prev = idx
        hyps.append(hyp)

    return hyps

def cer_stat(pre, gt):
    """
    Args:
        pre: list of token ids (模型预测)
        gt:  list of token ids (参考文本)
    Returns:
        cer: float, 字错率
        S, D, I, N: 替换数、删除数、插入数、目标长度
    """
    m, n = len(pre), len(gt)
    # dp[i][j] = (最小编辑距离, S, D, I)
    dp = [[(0, 0, 0, 0) for j in range(n+1)] for i in range(m+1)]
    # 初始化
    for i in range(1, m+1):
        dp[i][0] = (i, 0, 0, i)  # 全部插入
    for j in range(1, n+1):
        dp[0][j] = (j, 0, j, 0)  # 全部删除

    for i in range(1, m+1):
        for j in range(1, n+1):
            if pre[i-1] == gt[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                # 替换
                rep = dp[i-1][j-1]
                # 删除
                dele = dp[i][j-1]
                # 插入
                ins = dp[i-1][j]
                # 选最小
                candidates = [
                    (rep[0]+1, rep[1]+1, rep[2], rep[3]),       # 替换
                    (dele[0]+1, dele[1], dele[2]+1, dele[3]),   # 删除
                    (ins[0]+1, ins[1], ins[2], ins[3]+1),       # 插入
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    min_dist, S, D, I = dp[m][n]
    N = n  # 参考文本长度
    cer = (S+D+I) / (N+1e-8)
    return cer, S, D, I, N

def evaluate_ckpt(ckpt_path, test_dataloader, model, tokenizer, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    blank_id = tokenizer.blk_id()

    total_S, total_D, total_I, total_N = 0, 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Eval {os.path.basename(ckpt_path)}", ncols=120):
            batch = to_device(batch, device)
            audios = batch['audios']
            audio_lens = batch['audio_lens']
            texts = batch['texts']
            text_lens = batch['text_lens']
            
            # forward
            predict, _, out_lens = model(audios, audio_lens, texts, text_lens)
            # 解码
            pred_ids_batch = greedy_search(predict.cpu(), out_lens.cpu(), blank_id)  # List[List[int]]

            # 对每个样本计算CER
            for i in range(len(pred_ids_batch)):
                pred_ids = pred_ids_batch[i]
                tgt_ids = texts[i][:text_lens[i]].cpu().tolist()  # ground-truth token id序列
                _, S, D, I, N = cer_stat(pred_ids, tgt_ids)
                total_S += S
                total_D += D
                total_I += I
                total_N += N

    cer = (total_S + total_D + total_I) / (total_N + 1e-8)
    print(f"{os.path.basename(ckpt_path)} CER: {cer:.4f} (S={total_S}, D={total_D}, I={total_I}, N={total_N})")
    return cer

if __name__ == "__main__":
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer("./tokenizer/vocab.txt")
    model = Conformer_CTCModel(80, 256, tokenizer.size(), tokenizer.blk_id(), num_layers=6, nhead=8).to(device)

    test_dataloader = get_dataloader(
        "/data/mingxu/Audio/Assignment_3/dataset/split/test/wav.scp",
        "/data/mingxu/Audio/Assignment_3/dataset/split/test/pinyin",
        128, tokenizer, shuffle=False
    )

    ckpt_dir = "/data/mingxu/Audio/Assignment_3/ckpt"
    ckpt_list = [os.path.join(ckpt_dir, x) for x in os.listdir(ckpt_dir) if x.endswith(".pt")]
    ckpt_list.sort()  # 按照文件名排序

    log_path = "eval_conformer.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("ckpt\tCER\tS\tD\tI\tN\n")  # 表头
        for ckpt_path in ckpt_list:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state['model'])
            model.eval()
            blank_id = tokenizer.blk_id()
            total_S, total_D, total_I, total_N = 0, 0, 0, 0

            with torch.no_grad():
                for batch in tqdm(test_dataloader, desc=f"Eval {os.path.basename(ckpt_path)}", ncols=120):
                    batch = to_device(batch, device)
                    audios = batch['audios']
                    audio_lens = batch['audio_lens']
                    texts = batch['texts']
                    text_lens = batch['text_lens']
                    predict, _, out_lens = model(audios, audio_lens, texts, text_lens)
                    pred_ids_batch = greedy_search(predict.cpu(), out_lens.cpu(), blank_id)

                    for i in range(len(pred_ids_batch)):
                        pred_ids = pred_ids_batch[i]
                        tgt_ids = texts[i][:text_lens[i]].cpu().tolist()
                        _, S, D, I, N = cer_stat(pred_ids, tgt_ids)
                        total_S += S
                        total_D += D
                        total_I += I
                        total_N += N

            cer = (total_S + total_D + total_I) / (total_N + 1e-8)
            line = f"{os.path.basename(ckpt_path)}\t{cer:.6f}\t{total_S}\t{total_D}\t{total_I}\t{total_N}\n"
            print(line.strip())  # 仍然在屏幕打印一份
            f.write(line)