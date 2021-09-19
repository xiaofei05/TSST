def get_non_pad_mask(seq, pad):
    # seq: (B, L)
    assert seq.dim() == 2
    # (B, L)
    mask = seq.ne(pad).float()
    return mask.to(seq.device)

def get_seq_length(seq, pad):
    mask = get_non_pad_mask(seq, pad)
    # mask: (B, T)
    lengths = mask.sum(dim=-1)
    lengths = lengths.long()
    return lengths

def get_attn_mask(seq, pad):
    pad_mask = get_non_pad_mask(seq, pad)
    attn_mask = 1 - pad_mask
    attn_mask = attn_mask.bool()
    return attn_mask

def convert_ids_to_tokens(output_ids, vocab):
    outputs = []
    for sent in output_ids:
        tokens = []
        for word_id in sent:
            if word_id in vocab.special_tokens():
                continue
            else:
                tokens.append(vocab.id2word[word_id])
        outputs.append(" ".join(tokens))
    return outputs