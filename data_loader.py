import torch
from torch.utils.data import Dataset, DataLoader
import random

class Vocab(object):
    def __init__(self, vocab_path, add_special_tokens=True):
        
        self.word2id = {}
        self.id2word = []
        self.word_score = []

        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for token in special_tokens:
            self.word2id[token] = len(self.word2id)
            self.id2word.append(token)
            self.word_score.append(0)

        with open(vocab_path, "r", encoding="utf8") as f:
            for line in f:
                split_line = line.strip().split("\t")
                word, freqs = split_line[0], list(map(int, split_line[1:]))
                self.word2id[word] = len(self.word2id)
                self.id2word.append(word)
                
                # if there were only two styles, it would equal $1 - abs(freq0 - freq1) / (freq0 + freq1)$
                score = 0
                for freq in freqs:
                    score += abs((freq / sum(freqs)) - (1 / len(freqs)))
                self.word_score.append(1 - score)
        
        self.size = len(self.word2id)
        
        self.pad = self.word2id['<pad>']
        self.sos = self.word2id['<sos>']
        self.eos = self.word2id['<eos>']
        self.unk = self.word2id['<unk>']
    
    def __len__(self):
        return len(self.word2id)
    
    def special_tokens(self):
        return [self.pad, self.sos, self.eos, self.unk]

class LabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class DataIter:
    def __init__(self, dataloaders):
        
        self.loaders = dataloaders
        self.per_style_size = max([len(i) for i in self.loaders]) // len(self.loaders) + 1
        self.length = self.per_style_size * len(self.loaders)
        self.init_state()
        
    def __len__(self):
        return self.length
    
    def init_state(self):
        self.iters = [iter(i) for i in self.loaders]
        self.steps = []
        for i in range(len(self.loaders)):
            self.steps += [i] * self.per_style_size
        self.cur = 0
        random.shuffle(self.steps)
        
    def __next__(self):
        if self.cur < self.length:
            label_id = self.steps[self.cur]
            for out in self.iters[label_id]:
                self.cur += 1
                return out
        else:
            self.init_state()
            raise StopIteration

    def __iter__(self):
        self.init_state()
        return self


def collate_fn(batch, vocab, max_length, device):
    sents, labels = list(zip(*batch))
    max_len = min(max([len(s) for s in sents]), max_length)
    inputs = []

    for sent in sents:
        sent_ids = [vocab.word2id.get(w, vocab.unk) for w in sent][:max_len]
        inputs.append([vocab.sos] + sent_ids + [vocab.eos] + [vocab.pad] * (max_len - len(sent_ids)))
    
    return torch.LongTensor(inputs).to(device), torch.LongTensor(labels).to(device)


def get_dataloader(style_sents, vocab, batch_size, device, max_length=25, shuffle=False):
    dataloaders = []
    total_sents = []
    total_labels = []
    cfn = lambda batch: collate_fn(batch, vocab, max_length=max_length, device=device)
    
    for i, sents in enumerate(style_sents):
        labels = [i] * len(sents)
        total_sents += sents
        total_labels += labels
        dataset = LabelDataset(sents, labels)
        dataloaders.append(
            DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=cfn)
        )
    
    dataset = LabelDataset(total_sents, total_labels)
    pertrain_loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=cfn)
    train_loader = DataIter(dataloaders)
    return pertrain_loader, train_loader