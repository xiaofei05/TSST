import os
import torch
import torch.nn as nn
from model import Encoder, Criterion, ISRScheduler

class LM(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        
        self.word_embedding = nn.Embedding(len(vocab), config.embedding_size, padding_idx=vocab.pad)
        
        self.encoder = Encoder(config)
        self.h2word = nn.Linear(config.hidden_size, len(vocab))
        
    def forward(self, inputs):
        input_embeds = self.word_embedding(inputs)
        enc_outs, enc_state = self.encoder(input_embeds)
        
        batch_size, seq_len = enc_outs.size(0), enc_outs.size(1)
        outs = enc_outs.view(batch_size, seq_len, 2, -1)

        forward_outs = outs[:, :, 0, :] # [B, L, H]
        backward_outs = outs[:, :, 1, :]

        fouts = self.h2word(forward_outs)
        bouts = self.h2word(backward_outs)
        
        return fouts, bouts

def train_lm(config, log, vocab, model, train_loader):

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, betas=(0.9, 0.999), weight_decay=5e-5)
    optimizer = ISRScheduler(optimizer=optimizer, warmup_steps=config.warmup_steps,
            max_lr=config.max_lr, min_lr=config.min_lr, init_lr=config.init_lr, beta=0.75)

    seq_crit = Criterion(vocab.pad)
    
    out_dir = os.path.join(config.ckpts_dir, "pretrain_lm")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    model.train()

    log.info("Begin pretraining ...")
    for epoch in range(config.pretrain_epochs):
        for step, batch in enumerate(train_loader):
            inputs, labels = batch
            f_logits, b_logits = model(inputs)
            
            f_loss = seq_crit(f_logits, inputs[:, 1:], True)
            b_loss = seq_crit(b_logits, inputs[:, 1:], True)
            
            loss = (f_loss + b_loss) / 2.0            
            if step % 200==0:
                log.info("Pretrain Epoch {}, Step {}/{}, loss: {:.2f}, lr: {:.5f}".format(epoch+1, step+1, len(train_loader), loss.item(), optimizer.rate()))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        out_file = os.path.join(out_dir, "epoch{}.pt".format(epoch+1))
        torch.save(model.state_dict(), out_file)
    return

def load_pretrain(ckpt_dir, model_G, log):

    lm_state_dict = torch.load(ckpt_dir)
    gen_state_dict = model_G.state_dict()

    overlap_dic = {}
    for lm_key in lm_state_dict.keys():
        for g_key in model_G.state_dict().keys():
            if lm_key.split(".")[-2:] == g_key.split(".")[-2:]:
                overlap_dic[g_key] = lm_state_dict[lm_key]
                log.info("load %s from %s" % (g_key, lm_key))
    
    gen_state_dict.update(overlap_dic)
    model_G.load_state_dict(gen_state_dict)
    log.info("Loading pretrained params done!")


def pretrain(config, log, vocab, model_G, train_loader, load_ckpt=True):

    if load_ckpt:
        try:
            pretrain_file = os.path.join(config.ckpts_dir, "pretrain_lm/epoch{}.pt".format(config.pretrain_epochs))
            load_pretrain(pretrain_file, model_G, log)
            return
        except Exception as e:
            log.info("Loading pretrained lm model error: {}".format(e))
            pass

    lm_model = LM(config, vocab).to(config.device)
    train_lm(config, log, vocab, lm_model, train_loader)
    pretrain(config, log, vocab, model_G, train_loader, load_ckpt=True)