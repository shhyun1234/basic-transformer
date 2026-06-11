import torch
from torch.amp import autocast
from tqdm import tqdm
# output sequence [B, T, D]
# embedding vector [B, T, D]


def score_masking(seq, padding_mask):
    PAD_ID = 0
    BOS_ID = 2
    EOS_ID = 3
    
    return (seq == EOS_ID) | (seq == BOS_ID) | padding_mask

def BERTscore(model, ref: torch.Tensor, can: torch.Tensor, ref_padding_mask: torch.Tensor, can_padding_mask:torch.Tensor):
    # Encoder-Decoder 구조이기 때문에 hidden state가 아닌 embedding 비교 / 적절한 선택인건가?   
    # token이 True인 mask
    ref_mask = score_masking(ref, ref_padding_mask)
    can_mask = score_masking(can, can_padding_mask)
    
    # -100 -> 0 / 같이 하는 김에 실제 단어 토큰 외 전부 pad 처리
    ref = ref.masked_fill(ref_padding_mask, 0)
    can = can.masked_fill(can_padding_mask, 0)
    
    h1 = model.getting_dec_embedding(ref)
    h2 = model.getting_dec_embedding(can)
    
    h1 = torch.nn.functional.normalize(h1, dim=-1)
    h2 = torch.nn.functional.normalize(h2, dim=-1)
    
    # [B, T_r, T_c]
    # cosine similarity
    sim = torch.bmm(h1, h2.transpose(1,2))
    
    # [B, T_r, 1]
    ref_mask_3d = ref_mask.unsqueeze(2)
    # [B, 1, T_c]
    can_mask_3d = can_mask.unsqueeze(1)
    
    # before max padding
    sim = sim.masked_fill(ref_mask_3d, float('-inf'))
    sim = sim.masked_fill(can_mask_3d, float('-inf'))
    
    precision_score = sim.max(dim=1).values
    precision_score = precision_score.masked_fill(can_mask, 0.0)
    precision = precision_score.sum(dim=1) / (~can_mask).sum(dim=1)
    
    recall_score = sim.max(dim=2).values
    recall_score = recall_score.masked_fill(ref_mask, 0.0)
    recall = recall_score.sum(dim=1) / (~ref_mask).sum(dim=1)
    
    # [B, 1]
    f1score = 2*(precision * recall) / (precision + recall + 1e-8)
    
    return f1score

@torch.no_grad()
def create_candidate(model, batch, device, max_len):
    PAD_ID = 0
    BOS_ID = 2
    EOS_ID = 3
    
    forms = batch['forms'].to(device, non_blocking=True)
    tags = batch['tags'].to(device, non_blocking=True)
    enc_mask = batch['encoder_mask'].to(device, non_blocking=True)
    
    batch_size = forms.size(0)
    can = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    dec_in = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    model.init_kv_cache(batch_size, forms, tags, device, enc_key_padding_mask=enc_mask)
    for i in range(max_len-1):
        
        output, _ = model.inference(
            dec_in=dec_in,
            enc_key_padding_mask=enc_mask,
        )
        
        dec_in = output.argmax(dim=-1)
        dec_in[finished] = PAD_ID
        can = torch.concat([can, dec_in], dim=1)
        finished |= (dec_in.squeeze(1) == EOS_ID)
        
        if finished.all():
            break
        
    can = can[:,1:]
    can_padding_mask = (can == PAD_ID)


        
    return can, can_padding_mask

def valid_model_pass(model, batch, device, criterion, vocab_size):
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
    

@torch.no_grad()
def model_validation(model, valid_dataloader, device, criterion=None, vocab_size=None):

    model.eval()
    total_loss = torch.zeros((), device=device)
    
    for batch in tqdm(valid_dataloader):
        loss = valid_model_pass(model, batch, device, criterion, vocab_size)
        total_loss += loss.detach()
    return (total_loss / len(valid_dataloader)).item()
    
