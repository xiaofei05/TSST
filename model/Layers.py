import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.nlayers,
            dropout=config.dropout if config.nlayers > 1 else 0,
            bidirectional=(config.bidirectional==True),
            batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)

        
    def forward(self, inputs, lengths=None):
        # inputs: (B, L, emb_dim)
        # lengths: (B)
        input_embeds = self.dropout(inputs)

        if lengths is None:
            outputs, (state_h, state_c) = self.rnn(input_embeds, None)
        else:
            # Dynamic RNN
            packed = torch.nn.utils.rnn.pack_padded_sequence(input_embeds, lengths, batch_first=True, enforce_sorted=False)
            outputs, (state_h, state_c) = self.rnn(packed, None)
            # outputs: (B, L, 2*H)
            # state: (num_layers*num_directions, B, H)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, (state_h, state_c)

class Attention(nn.Module):
    def __init__(self, query_size, value_size, dropout=0.0):

        super(Attention, self).__init__()
        self.attn = nn.Linear(query_size + value_size, value_size)
        self.v = nn.Parameter(torch.rand(value_size))
        stdv = 1. / np.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attention_mask):
        # Q: (B, 1, H)
        # V: (B, L, num_directions * H)
        # attention_mask: (B, L), True means mask
        k_len = K.size(1)
        q_state = Q.repeat(1, k_len, 1) # (B, L, query_size)

        attn_energies = self.score(q_state, K) # (B, L)

        attn_energies.masked_fill_(attention_mask, -1e12)

        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights.bmm(V)
        return context, attn_weights

    def score(self, query, memory):
        energy = torch.tanh(self.attn(torch.cat([query, memory], 2)))
        energy = energy.transpose(1, 2)

        v = self.v.repeat(memory.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy) 
        return energy.squeeze(1)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(config.dropout)
        bidirectional = 2 if config.bidirectional else 1
        
        self.attention = Attention(
            query_size=config.hidden_size, 
            value_size=config.hidden_size*bidirectional,
            dropout=config.attention_dropout
        )

        self.attention2 = Attention(
            query_size=config.hidden_size, 
            value_size=config.hidden_size*bidirectional,
            dropout=config.attention_dropout
        )

        self.rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.nlayers,
            dropout=config.dropout if config.nlayers > 1 else 0,
            batch_first=True
        )
        self.q_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.style_linear = nn.Linear(config.hidden_size*bidirectional, config.hidden_size*bidirectional)
        self.linear = nn.Sequential(
            nn.Linear(config.embedding_size + config.hidden_size*bidirectional + config.hidden_size*bidirectional, 256),
            nn.Tanh()
        )

    def forward(self, input_embeds, last_state, enc_outs, attention_mask, style_features):
        # input_embeds: (B, 1, emb_size)
        embedded = self.dropout(input_embeds)

        # use h_t as query
        query = last_state[0][-1:].transpose(0,1) # (B, 1, H)
        
        # context: (B, 1, H*2)
        context, attn_weights = self.attention(query, enc_outs, enc_outs, attention_mask)
        
        style_mask = torch.zeros(style_features.size(0), style_features.size(1)).to(attention_mask.device).bool()
        style_features_trans = self.style_linear(style_features)
        style_context, _ = self.attention2(self.q_linear(query), style_features_trans, style_features_trans, style_mask)

        rnn_input = torch.cat([embedded, context, style_context], 2)
        x = self.linear(rnn_input)

        output, state = self.rnn(x, last_state)
        output = output.squeeze(1)  # (B, 1, N) -> (B, N)
        return output, state, attn_weights