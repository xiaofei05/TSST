import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random

class DenseRetriever(nn.Module):
    
    def __init__(self, unpaired_sents, vocab, add_sos=True, add_eos=True):
        super().__init__()
        
        self.pad_id = vocab.pad
        self.eos_id = vocab.eos
        self.sos_id = vocab.sos
        
        self.unpaired_sents, self.unpaired_ids = self.build_data(unpaired_sents, vocab, add_sos=add_sos, add_eos=add_eos)
        self.representations = None
        self.style_classes = len(self.unpaired_sents)
    
    def build_data(self, sents, vocab, add_sos, add_eos):
        print("Building Dense Retriever...")
        
        unpaired_ids = []
        unpaired_sents = []
        record_ids = set()
        for style_sents in sents:
            style_ids = []
            style_tokens = []
            for i in range(len(style_sents)):
                cur_sent = [vocab.word2id.get(word, vocab.unk) for word in style_sents[i]]
                if add_sos:
                    cur_sent = [self.sos_id] + cur_sent
                if add_eos:
                    cur_sent = cur_sent + [self.eos_id]
                cur_id = " ".join([str(s) for s in cur_sent])
                
                if cur_id not in record_ids:
                    style_ids.append(cur_id)
                    style_tokens.append(cur_sent)
                    record_ids.add(cur_id)

            unpaired_ids.append(style_ids)
            unpaired_sents.append(style_tokens)
        
        print("Total unique sentence: {}, {}".format(len(unpaired_sents[0]), len(unpaired_sents[1])))
        return unpaired_sents, unpaired_ids
        
    def update_representation(self, encoder, batch_size=32, device="cuda"):
        representations = []

        with torch.no_grad():
            for style_sents in self.unpaired_sents:
                style_reps = []
                for i in tqdm(range(0, len(style_sents), batch_size)):
                    batch = self.process_batch(style_sents[i:i+batch_size]).to(device)
                    batch_reps = encoder(batch).detach()
                    style_reps.append(batch_reps)
                representations.append(torch.cat(style_reps, dim=0))
        
        self.representations = [rep / rep.norm(p=2, dim=-1, keepdim=True) for rep in representations]
    
    
    def process_batch(self, sents):
        max_len = max([len(sent) for sent in sents])
        batch = []
        for sent in sents:
            batch.append(sent + [self.pad_id] * (max_len - len(sent)))
        return torch.LongTensor(batch)
        
    def process_retrieve_outs(self, batch_samples, batch_style):
        batch_sample_tokens = []
        for i in range(len(batch_samples)):
            for k in batch_samples[i]:
                sent = self.unpaired_sents[batch_style[i]][k]
                batch_sample_tokens.append(sent)
        
        batch_samples = self.process_batch(batch_sample_tokens).to(batch_style.device)
        # B, K, seq_length
        return batch_samples.view(batch_style.size(0), -1, batch_samples.size(-1))


    def retrieve(self, batch_inputs, batch_query, batch_style, topk=5):
        
        batch_size = batch_inputs.size(0)
        batch_ids = [" ".join([str(s) for s in seq if s != self.pad_id]) for seq in batch_inputs.cpu().numpy()]

        batch_query = batch_query / batch_query.norm(p=2, dim=-1, keepdim=True)
        batch_samples = np.zeros((batch_size, topk), dtype=np.int)
        
        for style in range(self.style_classes):
            
            style_index = (batch_style == style).nonzero().squeeze(dim=-1)
            if not style_index.nelement():
                continue
            
            sub_query = torch.index_select(batch_query, 0, style_index)
        
            scores = torch.matmul(sub_query, self.representations[style].t().contiguous())
            # (sub_batch, k+1)
            scores, index = torch.sort(scores, dim=-1, descending=True)
            
            # remove trival sentence
            for sub_cursor, batch_cursor in enumerate(style_index):
                z = 0
                for j in range(topk):
                    while(self.unpaired_ids[style][index[sub_cursor][z]] == batch_ids[batch_cursor]):
                        z += 1
                    batch_samples[batch_cursor][j] = index[sub_cursor][z]
                    z += 1
        return self.process_retrieve_outs(batch_samples, batch_style)