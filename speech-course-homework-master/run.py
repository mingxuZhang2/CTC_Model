import os
from data.dataloader import get_dataloader
from tokenizer.tokenizer import Tokenizer
from model.model import CTCModel
import torch
from utils.utils import to_device
import time
from tqdm import tqdm  

device = "cuda:3" if torch.cuda.is_available() else "cpu"
epochs = 30
accum_steps = 1
grad_clip = 5
log_file = "./train_log.txt"

tokenizer = Tokenizer()
model = CTCModel(80, 256, tokenizer.size(), tokenizer.blk_id()).to(device)
optim = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0005, betas=[0.9, 0.98], eps=1.0e-9,
    weight_decay=1.0e-6, amsgrad=False
)

train_dataloader = get_dataloader(
    "/data/mingxu/Audio/Assignment_3/dataset/split/train/wav.scp", "/data/mingxu/Audio/Assignment_3/dataset/split/train/pinyin", 128, tokenizer, shuffle=True)
test_dataloader = get_dataloader(
    "/data/mingxu/Audio/Assignment_3/dataset/split/test/wav.scp", "/data/mingxu/Audio/Assignment_3/dataset/split/test/pinyin", 128, tokenizer, shuffle=False)

if os.path.exists(log_file):
    print(f"Log file {log_file} exists, appending new logs.")
else:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("epoch,step,train_loss,valid_loss,lr,time\n")

for epoch in range(epochs):
    total_loss = 0.
    step_count = 0
    model.train()
    epoch_start = time.time()
    debug_flag = False
    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=130) as pbar:
        for i, input in pbar:
            step_count += 1
            input = to_device(input, device)
            audios = input['audios']
            audio_lens = input['audio_lens']
            texts = input['texts']
            text_lens = input['text_lens']
            debug_msgs = []
            if i < 3:  # 只检查前几个batch
                print("==== DEBUG BATCH", i, "====")
                print("audios stats:", audios.min().item(), audios.max().item(), audios.mean().item(), audios.std().item())
                print("audio_lens:", audio_lens)
                print("texts[0:5]:", texts[:5])
                print("text_lens:", text_lens)
                for name, param in model.named_parameters():
                    if "weight" in name:
                        print(f"{name}: min={param.data.min().item()}, max={param.data.max().item()}, mean={param.data.mean().item()}")
                        break
            if torch.isnan(audios).any() or torch.isinf(audios).any():
                debug_msgs.append(f"[!] audios has nan/inf at step {i}")
            if (audio_lens <= 0).any():
                debug_msgs.append(f"[!] audio_lens <= 0 at step {i}")
            if (text_lens <= 0).any():
                debug_msgs.append(f"[!] text_lens <= 0 at step {i}")
            if audios.min() < -1e5 or audios.max() > 1e5:
                debug_msgs.append(f"[!] audio feature abnormal range: min {audios.min().item():.2f}, max {audios.max().item():.2f}")
            if debug_msgs:
                print("\n".join(debug_msgs))
                debug_flag = True
            predict, loss, _ = model(audios, audio_lens, texts, text_lens)

            if torch.isnan(predict).any() or torch.isinf(predict).any():
                print(f"[!] predict has nan/inf at step {i}")
                debug_flag = True
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] loss is nan/inf at step {i}")
                debug_flag = True
            loss = loss / accum_steps
            total_loss += loss.item()
            loss.backward()
            if (i+1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                optim.zero_grad()
            
            avg_loss = total_loss / step_count
            elapsed = time.time() - epoch_start
            lr = optim.state_dict()['param_groups'][0]['lr']
            postfix_info = dict(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", time=f"{elapsed:.1f}s")
            if debug_flag:
                postfix_info['DEBUG'] = "Y"
            pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            pbar.set_postfix(postfix_info)
            debug_flag = False  # reset

            # 每10步写一次log
            if (i+1) % (accum_steps*10) == 0:
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{epoch+1},{i+1},{avg_loss},, {lr},{now}\n")

    model.eval()
    valid_loss = 0.
    valid_count = 0
    with torch.no_grad():
        for input in tqdm(test_dataloader, desc=f"Eval {epoch+1}", ncols=130):
            input = to_device(input, device)
            audios = input['audios']
            audio_lens = input['audio_lens']
            texts = input['texts']
            text_lens = input['text_lens']
            # ====== Debug: 输入检查 ======
            if torch.isnan(audios).any() or torch.isinf(audios).any():
                print(f"[!] (Eval) audios has nan/inf")
            if (audio_lens <= 0).any():
                print(f"[!] (Eval) audio_lens <= 0")
            if (text_lens <= 0).any():
                print(f"[!] (Eval) text_lens <= 0")
            _, loss, _ = model(audios, audio_lens, texts, text_lens)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[!] (Eval) loss is nan/inf")
            valid_loss += loss.item()
            valid_count += 1
    avg_train_loss = total_loss / step_count
    avg_valid_loss = valid_loss / (valid_count or 1)
    epoch_end = time.time()
    duration = epoch_end - epoch_start
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\n[{now}] Epoch {epoch+1} Done | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Time: {duration:.2f}s\n")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{epoch+1},end,{avg_train_loss},{avg_valid_loss},{lr},{now}\n")
    
    # 保存模型
    dict1 = {
        "model": model.state_dict(),
        "optimizer": optim.state_dict(),
    }
    torch.save(dict1, f"/data/mingxu/Audio/Assignment_3/ckpt/model_{epoch+1}.pt")