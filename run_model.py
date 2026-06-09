import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
from module.model_build import *
from module.train import model_train
from module.setup_data import load_dataloader

def main():
    #local_rank = int(os.environ["LOCAL_RANK"])
    #torch.cuda.set_device(local_rank)
    device = torch.device("cuda")


    # ko max token = 2807
    # en max token = 1371
    # cutting to 70
    # train 10784556, validation 988952

    EPOCHS = 4
    FORM_VOCAB_SIZE = 30000
    TAG_VOCAB_SIZE = 60
    DEC_VOCAB_SIZE = 30000
    MAX_LEN = 150
    EMBED_DIM = 256
    N_HEADS = 8
    N_LAYERS = 6
    DROPOUT = 0.1
    FFN_DIM = 512
    FFN_DIM = 512
    LEARNING_RATE = 3e-4

    enc_embedding = EncoderEmbeddingLayer(form_vocab_size=FORM_VOCAB_SIZE, tag_vocab_size=TAG_VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM, dropout=DROPOUT)
    dec_embedding = DecoderEmbeddingLayer(vocab_size=DEC_VOCAB_SIZE, max_len=MAX_LEN, embed_dim=EMBED_DIM, dropout=DROPOUT)

    encoder = Encoder(embed_dim=EMBED_DIM, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT, max_len=MAX_LEN)
    decoder = Decoder(embed_dim=EMBED_DIM, output_dim=FORM_VOCAB_SIZE, n_layers=N_LAYERS, n_heads=N_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT, embedding=dec_embedding, max_len=MAX_LEN, attn_kv_cache=False, cross_kv_cache=False)

    transformer = Transformer(encoder=encoder, decoder=decoder, enc_embedding=enc_embedding, dec_embedding=dec_embedding)
    transformer = transformer.to(device)
    transformer = torch.compile(transformer, dynamic=True)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    epoch_steps = 42_128
    warmup_steps = 10_000
    total_steps = epoch_steps * EPOCHS

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    from datasets import load_from_disk
    dataset = load_from_disk("data/koen_dataset")
    keep_columns = ['form_tokens', 'tag_tokens', 'en_tokens']
    for split in dataset:
        remove_columns = [c for c in dataset[split].column_names if c not in keep_columns]
        dataset[split] = dataset[split].remove_columns(remove_columns)
    dataset.set_format(type='torch', columns=keep_columns)

    train_dataloader, valid_dataloader = load_dataloader(dataset)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f"The model has {count_parameters(transformer):,} trainable parameters")


    def init_transformer_weights(m):
        # Linear layers (Attention, FFN 포함)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Embedding layer
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # LayerNorm
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    transformer.apply(init_transformer_weights)


    avg_loss = model_train(transformer, train_dataloader, valid_dataloader, optimizer, device, criterion=criterion, epochs=10, vocab_size = FORM_VOCAB_SIZE, scheduler=scheduler, accumulation_step=4, max_steps=epoch_steps)


    print(avg_loss)

if __name__ == '__main__':
    main()
