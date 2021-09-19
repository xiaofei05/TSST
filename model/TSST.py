import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from .Layers import Encoder, Decoder
from .utils import get_seq_length, get_attn_mask

class TSST(nn.Module):
    
    def __init__(self, config, vocab, retrieval):
        super().__init__()

        self.config = config
        self.vocab = vocab 
        
        # word embedding and weights
        self.word_embedding = nn.Embedding(len(vocab), config.embedding_size, padding_idx=vocab.pad)
        self.word_weights = nn.Embedding(len(vocab), 1)
        self.word_weights.weight.data.copy_(torch.Tensor(vocab.word_score).to(self.config.device).unsqueeze(1))

        self.content_enc = Encoder(config)
        self.style_enc = Encoder(config)
        self.dec = Decoder(config)
        
        self.h2word = nn.Linear(config.hidden_size, len(vocab))

        bidirectional = 2 if config.bidirectional else 1
        self.enc2dec_h = nn.Sequential(nn.Linear(config.hidden_size*bidirectional, config.hidden_size), nn.Tanh())
        self.enc2dec_c = nn.Sequential(nn.Linear(config.hidden_size*bidirectional, config.hidden_size), nn.Tanh())

        self.enc2q = nn.Sequential(nn.Linear(config.hidden_size*bidirectional, config.hidden_size*bidirectional), nn.Tanh())
        self.retrieval = retrieval
    

    def style_encode(self, inputs, query, labels):
    
        # B, K, max_lengths
        # batch_samples = labels
        batch_samples = self.retrieval.retrieve(inputs, query, labels, topk=self.config.K)
        # B*K, max_lengths
        batch_samples = batch_samples.view(-1, batch_samples.size(-1))
        
        lengths = get_seq_length(batch_samples, self.vocab.pad)
        input_embeds = self.word_embedding(batch_samples)
        enc_outs, enc_state = self.content_enc(input_embeds, lengths)
        
        # enc_h = enc_outs.sum(dim=1) / lengths.unsqueeze(1).float()
        enc_h, enc_c = self.get_encode_state(enc_state)
        # B, K, hidden*2
        enc_h = enc_h.view(inputs.size(0), -1, enc_h.size(-1))
        return enc_outs, enc_h, batch_samples

    def get_word_embedding(self, inputs):
        
        if inputs.dim() == 3:
            return torch.matmul(F.softmax(inputs, dim=-1), self.word_embedding.weight)
        else:
            return self.word_embedding(inputs)
    
    
    def update_retrieval(self):
        self.retrieval.update_representation(self.get_queries, self.config.batch_size, self.config.device)
        
    def get_queries(self, batch):
        attn_mask = get_attn_mask(batch, self.vocab.pad)
        enc_outs, enc_state = self.content_encode(batch)

        word_weights = self.word_weights(batch)
        word_weights.masked_fill_(attn_mask.unsqueeze(-1), -1e12)
        word_weights = F.softmax(word_weights, dim=1)

        query = (word_weights * enc_outs).sum(dim=1)
        return query 
    
    def get_encode_state(self, enc_state):
        bsz = enc_state[0].size(1)
        if self.config.bidirectional:
            enc_state_h = enc_state[0][-2:].transpose(0, 1).contiguous().view(bsz, -1)
            enc_state_c = enc_state[1][-2:].transpose(0, 1).contiguous().view(bsz, -1)
        else:
            enc_state_h = enc_state[0][-1]
            enc_state_c = enc_state[1][-1]
        return enc_state_h, enc_state_c



    def init_dec_state(self, enc_state):
        
        enc_state_h, enc_state_c = self.get_encode_state(enc_state)
        
        init_state_h = self.enc2dec_h(enc_state_h).unsqueeze(0)
        init_state_c = self.enc2dec_c(enc_state_c).unsqueeze(0)
        
        init_state = [
            init_state_h.repeat(self.config.nlayers, 1, 1), 
            init_state_c.repeat(self.config.nlayers, 1, 1)
        ]

        return init_state
    
    
    def content_encode(self, inputs):
        lengths = get_seq_length(inputs, self.vocab.pad)
        input_embeds = self.word_embedding(inputs)
        enc_outs, enc_state = self.content_enc(input_embeds, lengths)
        
        return enc_outs, enc_state
    
    def decode_step(self, input, last_state, enc_outs, attn_mask, style_features):
        input_embed = self.word_embedding(input)
        out, state, _ = self.dec(input_embed, last_state, enc_outs, attn_mask, style_features)
        logits = self.h2word(out)
        return logits, state
        

    def decode(self, enc_outs, last_state, attn_mask, style_features, targets=None):
        batch_size = enc_outs.size(0)
        if targets is not None:
            target_len = targets.size(1)
        
        output_logits = []
        input = torch.zeros(batch_size, 1, dtype=torch.long, device=enc_outs.device).fill_(self.vocab.sos)

        for t in range(self.config.max_length+1):
            logits, last_state = self.decode_step(input, last_state, enc_outs, attn_mask, style_features)
            output_logits.append(logits.unsqueeze(1))            
            is_teacher = random.random() < self.config.teacher_forcing_ratio
            
            if (targets is not None) and (t < target_len) and is_teacher:
                input = targets[:, t].unsqueeze(1)
            else:
                input = torch.argmax(logits, dim=-1).unsqueeze(1)
        
        output_logits = torch.cat(output_logits, dim=1)
        return output_logits
    
    def forward(self, inputs, labels, targets=None):
        
        attn_mask = get_attn_mask(inputs, self.vocab.pad)
        enc_outs, enc_state = self.content_encode(inputs)

        dec_state = self.init_dec_state(enc_state)

        # get style-independent query embedding
        word_weights = self.word_weights(inputs)
        word_weights.masked_fill_(attn_mask.unsqueeze(-1), -1e12)
        word_weights = F.softmax(word_weights, dim=1)
        query = (word_weights * enc_outs).sum(dim=1)
        
        _, style_features, style_samples = self.style_encode(inputs, query, labels)
        
        output_logits = self.decode(enc_outs, dec_state, attn_mask, style_features, targets)

        return output_logits, (query, style_samples)
