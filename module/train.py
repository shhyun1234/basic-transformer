import torch
from tqdm import tqdm
import time
from torch.amp import autocast
from evaluate import model_validation

# 1. vram 한계로 인한 mixed precision, step 학습 추가
# 2. 그래픽카드 병렬 사용을 위해 나중에 DDP 추가

def epoch_time(start_time, end_time):
    total_time = end_time - start_time
    total_mins = int(total_time / 60)
    total_secs = int(total_time - (total_mins * 60))
    return total_mins, total_secs


def train_model_pass(model, batch, device, criterion, vocab_size):
    # non_blocking 추가, 근데 다음 배치를 미리 준비하는 구조가 아니기 때문에 실효성은??
    # gpu 사용량을 보고 prefetch 구조 도입
    forms = batch['forms'].to(device, non_blocking=True)
    tags = batch['tags'].to(device, non_blocking=True)
    dec_in = batch['decoder_input_ids'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)
    enc_mask = batch['encoder_mask'].to(device, non_blocking=True)
    dec_mask = batch['decoder_mask'].to(device, non_blocking=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        output, _ = model(
            forms=forms,
            tags=tags,
            dec_in=dec_in,
            enc_key_padding_mask=enc_mask,
            dec_key_padding_mask=dec_mask
        )
        # output [B, T, D], label [B, T] class
        # loss 계산
        # torch.nn.CrossEntropy는 내부적으로 softmax 적용 후에 비교 => logit을 건네줘야 함
        loss = criterion(
            output.view(-1, vocab_size),
            labels.view(-1)
        )
    return loss
    


def model_train(model, train_dataloader, valid_dataloader, optimizer, device, criterion=None, epochs=4, vocab_size = None, scheduler=None,
                sampler=None, accumulation_step=4, max_steps=2000):
    
    assert vocab_size != None, 'vocab_size 입력 필요'
    
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val = 10

    for epoch in range(epochs):
        
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        epoch_step = 0
        model.train()
        epoch_loss = torch.zeros((), device=device)

        loader_iter = tqdm(train_dataloader)
        
        optimizer.zero_grad() # step 학습으로 루프 이동
        
        for step, batch in enumerate(loader_iter):
            loss = train_model_pass(model, batch, device, criterion, vocab_size)
            (loss / accumulation_step).backward()
            
            
            if (step+1) % accumulation_step == 0:
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                # step마다 zero_grad
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()
                
                epoch_step += 1
                
                # batch loop 탈출
                if max_steps is not None and epoch_step >= max_steps:
                    break

            epoch_loss += loss.detach()
        
        # 남은 gradient 처리
        if (step + 1) % accumulation_step != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        
        avg_loss = (epoch_loss / (step + 1)).item()
        
        end_time = time.time()
        epochs_mins, epochs_secs = epoch_time(start_time, end_time)
        
        val_loss = model_validation(model, valid_dataloader, device,criterion=criterion, vocab_size=vocab_size)

        print(f"Epoch {epoch+1} | Time: {epochs_mins}m {epochs_secs}s | Train Loss: {avg_loss:.4f} | Valid Loss:{val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, avg_loss, f"checkpoint/checkpoint_epoch_{epoch}.pt")
        
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, f"checkpoint/best_model.pt")
                    

    save_checkpoint(model, optimizer, epoch, avg_loss,  "checkpoint/final_model.pt")
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, avg_loss, path):   
    torch.save({
        "epoch": epoch,
        "loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
