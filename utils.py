import torch

def read_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(line.strip().lower())
    return data

def convert_ids_to_tokens(output_ids, vocab):
    outputs = []
    for sent in output_ids:
        tokens = []
        for word_id in sent:
            if word_id in [vocab.sos, vocab.unk, vocab.pad]:
                continue
            if word_id == vocab.eos:
                break
            else:
                tokens.append(vocab.id2word[word_id])
        outputs.append(" ".join(tokens))
    return outputs

def safe_loss(ori_loss):
    loss = torch.where(torch.isnan(ori_loss), torch.full_like(ori_loss, 0.0), ori_loss)
    loss = torch.where(torch.isinf(loss), torch.full_like(loss, 1.0), loss)
    return loss

def process_outputs(output_ids, eos_id, pad_id, sos_id):
    batch_size, _ = output_ids.size()
    sents = []
    for i in range(batch_size):
        sent = output_ids[i].cpu().tolist()
        if eos_id in sent:
            sent = sent[:sent.index(eos_id)]
        sent = [word for word in sent if word not in [eos_id, pad_id, sos_id]]
        sents.append(sent)
    
    max_len = max([len(i) for i in sents])
    inputs = [[sos_id] + sent + [eos_id] + [pad_id]*(max_len - len(sent)) for sent in sents]
    return torch.LongTensor(inputs).to(output_ids.device)


def get_bow_labels(inputs, samples, vocab):
    vocab_size = len(vocab)
    inputs_list = inputs.cpu().tolist()
    sample_list = samples.cpu().tolist()
    bow_one_hot_labels = []
    for i in range(len(inputs_list)):
        orgin_set = set(inputs_list[i])
        retrieve_set = set()
        for j in sample_list[i]:
            retrieve_set = retrieve_set | set(j)
        retrieve_set = retrieve_set - set(vocab.special_tokens())
        bow_label = list(retrieve_set - orgin_set)

        bow_label = torch.LongTensor(bow_label)
        bow_one_hot = torch.zeros(1, vocab_size)
        bow_one_hot.index_fill_(1, bow_label, 1)
        bow_one_hot_labels.append(bow_one_hot)

    bow_one_hot_labels = torch.cat(bow_one_hot_labels, dim=0).to(inputs.device)
    return bow_one_hot_labels