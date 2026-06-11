import torch
from torch import nn

# kv cache 추가로 offset도 추가 / 왜 이런 중요한 부분을 놓치고 있었을까?
class PositionEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim, requires_grad=False)
        self.register_buffer("PE", pe)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, embed_dim, step=2, dtype=torch.float)
        
        div = 10000 ** (_2i / embed_dim)

        pe[:,0::2] = torch.sin(pos / div)
        pe[:,1::2] = torch.cos(pos / div)

    def forward(self, x, offset=0):
        seq_len = x.size(1)
        posen_x = self.PE[offset:offset + seq_len,:].unsqueeze(0)

        return posen_x

# Encoder와 Decoder Embedding 공유 --> 한국어랑 영어를 동시에 tokenize해야 함
# -> 한국어 kiwi, 영어 bpe로 tokenizer 변경, embedding 분리

class EncoderEmbeddingLayer(nn.Module):
    def __init__(self, form_vocab_size, tag_vocab_size, max_len, embed_dim, dropout):
        super().__init__()
        self.form_embedding = nn.Embedding(form_vocab_size, embedding_dim=embed_dim)
        self.tag_embedding = nn.Embedding(tag_vocab_size, embedding_dim=embed_dim)
        self.peencoder = PositionEncoding(max_len=max_len, embed_dim=embed_dim)
        self.scale = embed_dim ** 0.5
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, forms, tags):
        x = (self.form_embedding(forms) + self.tag_embedding(tags)) * self.scale + self.peencoder(forms)
        return self.dropout(x)

class DecoderEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.peencoder = PositionEncoding(max_len=max_len, embed_dim=embed_dim)
        self.scale = embed_dim ** 0.5
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, offset=0):
        return self.dropout(self.embedding(x) * self.scale + self.peencoder(x, offset))


# TODO maksing 부분 분기 축소 / self.mask는 고정이니까 causal mask는 괜찮을거 같고 padding mask를 고정, 


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, dropout=0.1, max_len=256, mask=False, attn_kv_cache=False, cross_kv_cache=False):
        super().__init__()

        assert embed_dim % n_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads
        self.mask = mask
        self.attn_kv_cache = attn_kv_cache
        self.cross_kv_cache = cross_kv_cache
        self.register_buffer('causal_mask', torch.tril(torch.ones(max_len, max_len)).bool())

        self.w_q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.w_k = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.w_v = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.w_o = nn.Linear(in_features=embed_dim, out_features=embed_dim)

        self.dropout = nn.Dropout(p=dropout)
        # nn.Module이 아닌 일반 Tensor이기 때문에 to(device) 해줘야함
        # self.scale = torch.sqrt(torch.tensor([self.head_dim], dtype=torch.float)).to(device)
        # 폐기. scalar 변수 하나로 처리
        self.scale = self.head_dim ** 0.5


    
    def forward(self, query, key, value, key_padding_mask=None, kv_cache=None):
        #input [batch_size, data_len, embed_dim]
        batch_size = query.shape[0]
        
        # shape: [batch_size, data_len, embed_dim]
        Q = self.w_q(query)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        if self.cross_kv_cache:
            K = key
            V = value
        
        else:
            K = self.w_k(key)
            V = self.w_v(value)

            # embed_dim -> n_heads x head_dim
            # shape: view [batch_size, data_len, n_heads, head_dim] [B, T, H, D] -> permute [batch_size, n_heads, data_len, head_dim] [B, H, T, D]
            K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
            V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        # permute는 연산 성능 손실이 존재하지만 몇 번의 계산만이라면 continguous로 재정렬하는 손실이 더 크다.

        if self.attn_kv_cache:
            t = kv_cache['length']
            # 차원 유지 용도 t:t+1
            kv_cache['attn_K'][:, :, t:t+1, :] = K
            kv_cache['attn_V'][:, :, t:t+1, :] = V
            kv_cache['length'] += 1
            K = kv_cache['attn_K'][:, :, :t+1, :]
            V = kv_cache['attn_V'][:, :, :t+1, :]
        # Attention energy
        # QK^T / sqrt(head_dim) -> [B, H, T_q, D] x [B, H, D, T_k] = [B, H, T_q, T_k]
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        
        B, H, T_q, d_k = Q.shape
        T_k = K.shape[2]

        mask = torch.zeros((B, 1, T_q, T_k), dtype=torch.bool, device=Q.device)

        # masking
        if self.mask:
            # key에 casual mask 적용
            # assert T_q == T_k, "cross attention에서 causal mask 사용안함"
            causal_mask = self.causal_mask[:T_q, :T_k].view(1, 1, T_q, T_k)
            mask = mask | ~(causal_mask)
        # padding masking
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k]
            # → [B, 1, 1, T_k] 로 확장
            key_padding_mask = key_padding_mask.view(B, 1, 1, T_k)
            mask = mask | key_padding_mask

        energy = energy.masked_fill(mask, float('-inf'))


        # Attention score
        # 각 head 마다 softmax
        attention = torch.softmax(energy, dim=-1)
        
        # 전체 행 -inf로 nan값 생성 시 처리
        attention = torch.nan_to_num(attention, nan=0.0)

        # attention에 dropout으로 특정 token 집중 방지
        # attention [B, H, T_q, T_k] ,V [B, H, T_v, D] T_k, T_v 동일
        # value의 weighted sum
        x = torch.matmul(self.dropout(attention), V)
        # [B, H, T_q, D] -> [B, T_q, H, D] -> [B, T_q, embed_dim]
        x = x.permute(0,2,1,3).contiguous().view(batch_size, -1, self.embed_dim)

        x = self.w_o(x)

        return x, attention

        
class AttentionFeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=ffn_dim, bias=True),
            nn.GELU(), # ReLU -> GELU
            nn.Dropout(p=dropout),
            nn.Linear(in_features=ffn_dim, out_features=embed_dim, bias=True)
        )

    def forward(self, x):
        return self.ffn(x)

# Encoder와 Decoder에 Pre-LN 적용

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout, max_len):
        super().__init__()
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

        self.self_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, max_len=max_len)
        self.ffn_layer = AttentionFeedForwardLayer(embed_dim=embed_dim, ffn_dim=ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, key_padding_mask=None):
        x_norm = self.attn_layer_norm(x)
        x_in, _ = self.self_attention(x_norm, x_norm, x_norm, key_padding_mask)
        x = self.dropout(x_in) + x

        x_norm = self.ffn_layer_norm(x)
        x_in = self.ffn_layer(x_norm)
        x = self.dropout(x_in) + x

        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, n_layers, n_heads, ffn_dim, dropout, max_len):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.encoder = nn.ModuleList([EncoderLayer(embed_dim=embed_dim, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout, max_len=max_len) 
                                             for _ in range(n_layers)])

    # PE encoding까지 거친 embedding 입력
    def forward(self, x, key_padding_mask=None):
        for layer in self.encoder:
            x = layer(x, key_padding_mask)

        return self.final_layer_norm(x)

# masking 추가
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout, max_len, use_kv_cache):
        super().__init__()
        self.masked_attn_norm = nn.LayerNorm(embed_dim)
        self.enc_attn_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        if use_kv_cache:
            self.masked_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, max_len=max_len, mask=False, attn_kv_cache=True)
        else:
            self.masked_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, max_len=max_len, mask=True)
        
        if use_kv_cache:
            self.enc_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, max_len=max_len, cross_kv_cache=True)
        else:
            self.enc_attention = MultiHeadAttentionLayer(embed_dim=embed_dim, n_heads=n_heads, dropout=dropout, max_len=max_len)
        
        self.ffn_layer = AttentionFeedForwardLayer(embed_dim=embed_dim, ffn_dim=ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, enc_src, enc_key_padding_mask=None, dec_key_padding_mask=None):
        # masked_attention
        trg_norm = self.masked_attn_norm(trg)
        trg_in, _ = self.masked_attention(trg_norm, trg_norm, trg_norm, dec_key_padding_mask)
        trg = self.dropout(trg_in) + trg

        # enc + dec attention
        trg_norm = self.enc_attn_norm(trg)
        trg_in, attention = self.enc_attention(trg_norm, enc_src, enc_src, enc_key_padding_mask)
        trg = self.dropout(trg_in) + trg

        # ffn
        trg_norm = self.ffn_norm(trg)
        trg_in = self.ffn_layer(trg_norm)
        trg = self.dropout(trg_in) + trg

        return trg, attention
    
    def inference(self, trg, kv_cache, enc_key_padding_mask=None):
        # masked_attention
        trg_norm = self.masked_attn_norm(trg)
        trg_in, _ = self.masked_attention(trg_norm, trg_norm, trg_norm, kv_cache=kv_cache)
        trg = self.dropout(trg_in) + trg

        # enc + dec attention
        trg_norm = self.enc_attn_norm(trg)
        trg_in, attention = self.enc_attention(trg_norm, kv_cache['cross_K'], kv_cache['cross_V'], enc_key_padding_mask)
        trg = self.dropout(trg_in) + trg

        # ffn
        trg_norm = self.ffn_norm(trg)
        trg_in = self.ffn_layer(trg_norm)
        trg = self.dropout(trg_in) + trg

        return trg, attention
class Decoder(nn.Module):
    def __init__(self, embed_dim, output_dim, n_layers, n_heads, ffn_dim, dropout, embedding, max_len, use_kv_cache):
        super().__init__()
        self.decoders = nn.ModuleList([DecoderLayer(embed_dim=embed_dim, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout, max_len=max_len, use_kv_cache=use_kv_cache) 
                                      for _ in range(n_layers)])
        self.linear = nn.Linear(embed_dim, output_dim, bias=False)
        self.linear.weight = embedding.embedding.weight # decoder input embedding과 weight tying
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.max_len = max_len

    # 모델 자체에 kv cache 분기를 만들어야 돼??? 아니 생성 시에 inference 용으로 생성하도록 인자로 구분
    def forward(self, trg, enc_src, enc_key_padding_mask=None, dec_key_padding_mask=None):
        for layer in self.decoders:
            trg, attention = layer(trg, enc_src, enc_key_padding_mask=enc_key_padding_mask, dec_key_padding_mask=dec_key_padding_mask)

        output = self.linear(self.final_layer_norm(trg))

        return output, attention
    
    def inference(self, trg, kv_cache, enc_key_padding_mask=None):
        for i, layer in enumerate(self.decoders):
            trg, attention = layer.inference(trg, kv_cache[i], enc_key_padding_mask)            

        output = self.linear(self.final_layer_norm(trg))
        
        return output, attention
    
    
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, enc_embedding, dec_embedding):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.kv_cache = None
        self.n_heads = decoder.n_heads
        self.head_dim = decoder.head_dim
        self.max_len = decoder.max_len

    # [B, H, T, D]
    def init_kv_cache(self, batch_size, forms, tags, device, enc_key_padding_mask=None):
        src = self.enc_embedding(forms, tags)
        enc_src = self.encoder(src, key_padding_mask=enc_key_padding_mask)
        self.kv_cache = [
            {
                'attn_K': torch.zeros(batch_size, self.n_heads, self.max_len, self.head_dim, device=device),
                'attn_V': torch.zeros(batch_size, self.n_heads, self.max_len, self.head_dim, device=device),
                'cross_K': layer.enc_attention.w_k(enc_src).view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3),
                'cross_V': layer.enc_attention.w_v(enc_src).view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3),
                'length': 0
            }
            for layer in self.decoder.decoders
        ]
    
    
    def forward(self, forms, tags, dec_in, enc_key_padding_mask=None, dec_key_padding_mask=None):
        src = self.enc_embedding(forms, tags)
        trg = self.dec_embedding(dec_in)

        enc_src = self.encoder(src, key_padding_mask=enc_key_padding_mask)

        output, attention = self.decoder(trg, enc_src, enc_key_padding_mask=enc_key_padding_mask, dec_key_padding_mask=dec_key_padding_mask)

        return output, attention

    def inference(self, dec_in, enc_key_padding_mask=None):
        pos = self.kv_cache[0]['length']
        trg = self.dec_embedding(dec_in, offset=pos)
        output, attention = self.decoder.inference(trg, self.kv_cache, enc_key_padding_mask=enc_key_padding_mask)
        
        return output, attention
    
    def getting_dec_embedding(self, seq):
        return self.dec_embedding(seq)

