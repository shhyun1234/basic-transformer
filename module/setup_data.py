import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# bucket sampling 추후 적용

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
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size = 256,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    valid_dataloader = DataLoader(
        dataset['validation'],
        batch_size = 256,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    return train_dataloader, valid_dataloader
