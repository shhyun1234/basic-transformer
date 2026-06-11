import torch
from module.model_build import *
from module.evaluate import create_candidate, BERTscore
from module.setup_data import load_dataloader
from tqdm import tqdm
import time, os, glob

def evaluate(pt_path):
    # 1번 gpu로 학습과 병렬로 evaluate
    device = torch.device("cuda:1")
    
    FORM_VOCAB_SIZE = 30000
    TAG_VOCAB_SIZE = 60
    DEC_VOCAB_SIZE = 30000
    MAX_LEN = 150
    EMBED_DIM = 384
    N_HEADS = 6
    N_LAYERS = 8
    DROPOUT = 0.05
    FFN_DIM = 1024
    
    enc_embedding = EncoderEmbeddingLayer(form_vocab_size=FORM_VOCAB_SIZE, tag_vocab_size=TAG_VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM, dropout=DROPOUT)
    dec_embedding = DecoderEmbeddingLayer(vocab_size=DEC_VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM, dropout=DROPOUT)
    
    encoder = Encoder(embed_dim=EMBED_DIM, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=MAX_LEN)
    decoder = Decoder(embed_dim=EMBED_DIM, output_dim=DEC_VOCAB_SIZE, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT, embedding=dec_embedding, max_len=MAX_LEN, use_kv_cache=True)
    
    transformer = Transformer(encoder=encoder, decoder=decoder, enc_embedding=enc_embedding, dec_embedding=dec_embedding)
    transformer = transformer.to(device)
    
    from datasets import load_from_disk
    dataset = load_from_disk("data/koen_dataset")
    keep_columns = ['form_tokens', 'tag_tokens', 'en_tokens']
    for split in dataset:
        remove_columns = [c for c in dataset[split].column_names if c not in keep_columns]
        dataset[split] = dataset[split].remove_columns(remove_columns)
    dataset.set_format(type='torch', columns=keep_columns)
    
    train_dataloader, valid_dataloader, train_sampler = load_dataloader(dataset)
    
    # torch.load는 저장 시점 gpu에 올리려고 시도함
    state = torch.load(pt_path, map_location=device)
    transformer.load_state_dict(state['model_state_dict'])
    transformer.eval()
    
    f1s = []
    
    for batch in tqdm(valid_dataloader):
        can, can_padding_mask = create_candidate(transformer, batch, device, MAX_LEN)
        ref_padding_mask = batch['labels'] == -100
        ref_padding_mask = ref_padding_mask.to(device)
        f1score = BERTscore(transformer, batch['labels'].to(device), can, ref_padding_mask, can_padding_mask)
        f1s.append(f1score.detach())
    
    total_f1score = torch.cat(f1s, dim=0).mean()
    
    return total_f1score


def evaluate_while_train():
    
    watch_dir = "./checkpoint"
    
    print("Watching Start...")
    
    while True:
        done_files = glob.glob(os.path.join(watch_dir, "*.done"))
        
        for done_path in done_files:
            pt_path = done_path.replace(".done", ".pt")
            
            if not os.path.exists(pt_path):
                continue
            
            print(f"[Evaluate] {pt_path}")
            
            total_f1score = evaluate(pt_path)
            print(f"[Evaluate] F1score: {total_f1score:.4f}")
            
            with open("checkpoint/f1score_log.txt", 'a', encoding='utf-8') as f:
                f.write(f"{done_path}, f1={total_f1score:.4f}\n")
            
            os.remove(done_path)
        
        time.sleep(60)

if __name__ == '__main__':
    evaluate_while_train()
