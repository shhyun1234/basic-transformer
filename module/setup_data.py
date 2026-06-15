import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler
import numpy as np

# bucket sampling 추후 적용

def compute_lengths(ds):
    # 번역 태스크는 src+tgt 합으로 버킷팅하면 패딩 절감 효과가 더 큼
    out = ds.with_format(None).map(lambda batch: {
        'length': [len(f) + len(e) for f, e in zip(batch['form_tokens'], batch['en_tokens'])]
    }, batched=True, batch_size=10_000, num_proc=8)
    return out['length']

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, pool_mult=100, shuffle=True, seed=42):
        self.lengths = lengths          # 샘플별 길이 리스트 (미리 계산)
        self.batch_size = batch_size
        self.pool_size = batch_size * pool_mult
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        lengths = np.asarray(self.lengths)
        
        # indices: index
        if self.shuffle:
            indices = rng.permutation(len(lengths))
        else:
            indices = np.arange(len(lengths))
        

        batches = []
        for i in range(0, len(indices), self.pool_size):
            pool = indices[i:i + self.pool_size]
            
            # 풀 내부만 정렬
            pool = pool[np.argsort(lengths[pool], kind='stable')]
            for j in range(0, len(pool), self.batch_size):
                batches.append(pool[j:j + self.batch_size].tolist())

        # 배치 순서 셔플
        if self.shuffle:
            rng.shuffle(batches)

        yield from batches

    def __len__(self):
        # 정수 형태 ceiling
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def collate_fn(batch):
    PAD_ID = 0
    BOS_ID = 2
    EOS_ID = 3

    forms = []
    tags = []
    decoder_inputs = []
    labels = []
    
    bos = torch.tensor([BOS_ID], dtype=torch.long)
    eos = torch.tensor([EOS_ID], dtype=torch.long)
    for x in batch:
        f = x['form_tokens']
        t = x['tag_tokens']
        en = x['en_tokens']

        dec_in = torch.cat([bos, en])
        lab = torch.cat([en, eos])

        forms.append(f)
        tags.append(t)
        decoder_inputs.append(dec_in)
        labels.append(lab)

    # padding
    forms = pad_sequence(forms, batch_first=True, padding_value=PAD_ID)
    tags = pad_sequence(tags, batch_first=True, padding_value=PAD_ID)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_ID)

    # label padding -100 -> pytorch CrossEntropyLoss default 옵션 label value = -100인 위치 loss 제외
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # mask
    encoder_padding_mask = (forms == PAD_ID)
    decoder_padding_mask = (decoder_inputs == PAD_ID)

    return {
        'forms': forms,
        'tags': tags,
        'decoder_input_ids': decoder_inputs,
        'labels': labels,
        'encoder_mask': encoder_padding_mask,
        'decoder_mask': decoder_padding_mask,
    }

def load_dataloader(dataset):
    train_lengths = compute_lengths(dataset['train'])
    valid_lengths = compute_lengths(dataset['validation'])
    
    train_sampler = BucketBatchSampler(train_lengths, batch_size=256, shuffle=True)
    valid_sampler = BucketBatchSampler(valid_lengths, batch_size=256, shuffle=False)
    
    train_dataloader = DataLoader(
        dataset['train'],
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        dataset['validation'],
        batch_sampler=valid_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, valid_dataloader, train_sampler
