import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LABEL_MAP
from model import Criterion, ISRScheduler
from utils import convert_ids_to_tokens, safe_loss, process_outputs, get_bow_labels

def train_generator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion):    
    inputs, labels = batch

    # rec loss
    rec_logits, (_, _) = model_G(inputs, labels, targets=inputs[:, 1:])
    rec_loss = seq_criterion(rec_logits, inputs[:, 1:], truncate=True)
    
    # cyc loss
    rev_logits, (rec_q, rev_samples) = model_G(inputs, 1-labels, targets=None)

    rev_inputs = process_outputs(torch.argmax(rev_logits, dim=-1), vocab.eos, vocab.pad, vocab.sos)
    cyc_logits, (rev_q, raw_samples) = model_G(rev_inputs, labels, targets=inputs[:, 1:])
    cyc_loss = seq_criterion(cyc_logits, inputs[:, 1:], truncate=True)
    
    # retrieval loss
    cosloss = nn.CosineEmbeddingLoss(reduction="none")
    targets = torch.ones(rev_q.size(0)).to(rev_q.device)
    q_loss = cosloss(rec_q, rev_q, targets)
    q_loss = safe_loss(q_loss)
    q_loss = q_loss.mean()

    # bow loss
    rev_samples_bow = rev_samples.view(inputs.size(0), config.K, -1) 
    raw_samples_bow = raw_samples.view(inputs.size(0), config.K, -1) 

    rev_bow_labels = get_bow_labels(inputs, rev_samples_bow, vocab).unsqueeze(1).repeat(1, rev_logits.size(1), 1).view(-1, len(vocab))
    cyc_bow_labels = get_bow_labels(rev_inputs, raw_samples_bow, vocab).unsqueeze(1).repeat(1, cyc_logits.size(1), 1).view(-1, len(vocab))
    
    rev_bow_labels_num = rev_bow_labels.sum(dim=-1)
    cyc_bow_labels_num = cyc_bow_labels.sum(dim=-1)

    rev_bow_loss =  (- F.log_softmax(rev_logits.view(-1, rev_logits.size(-1)), dim=-1) * rev_bow_labels).sum(dim=-1) / (rev_bow_labels_num+1e-2)
    rev_bow_loss = safe_loss(rev_bow_loss.view(inputs.size(0), -1))
    cyc_bow_loss =  (- F.log_softmax(cyc_logits.view(-1, cyc_logits.size(-1)), dim=-1) * cyc_bow_labels).sum(dim=-1) / (cyc_bow_labels_num+1e-2)
    cyc_bow_loss = safe_loss(cyc_bow_loss.view(inputs.size(0), -1))
    bow_loss = (rev_bow_loss.mean() + cyc_bow_loss.mean()) / 2

    # adv  loss 
    rev_input_embeds = model_G.get_word_embedding(rev_logits)
    class_logits = model_D(rev_input_embeds)
    adv_loss = cls_criterion(class_logits, 1-labels)

    losses = rec_loss + cyc_loss + adv_loss + q_loss + bow_loss
    losses = losses.mean()

    optim_G.zero_grad()
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model_G.parameters(), config.max_grad_norm)
    optim_G.step()
    
    return losses.item(), rec_loss.item(), cyc_loss.item(), adv_loss.item(), q_loss.item(), bow_loss.item()


def train_disriminator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion):
    
    inputs, labels = batch
    # rec
    with torch.no_grad():
        rec_logits, (rec_q, rec_samples) = model_G(inputs, labels, targets=inputs[:, 1:])
        rev_logits, (rev_q, rev_samples) = model_G(inputs, 1-labels, targets=None)
        rev_input_embeds = model_G.get_word_embedding(rev_logits)
        raw_input_embeds = model_G.get_word_embedding(rec_logits)
        golden_input_embedds = model_G.get_word_embedding(inputs[:, 1:])
        golden_rec_embedds = model_G.get_word_embedding(rec_samples[:, 1:])
        golden_rev_embedds = model_G.get_word_embedding(rev_samples[:, 1:])
    
    rev_loss = cls_criterion(model_D(rev_input_embeds), torch.zeros_like(labels).fill_(2))
    rec_loss = cls_criterion(model_D(raw_input_embeds), labels)
    raw_loss = cls_criterion(model_D(golden_input_embedds), labels)

    sample_raw_loss = cls_criterion(model_D(golden_rec_embedds), labels.repeat(config.K))
    sample_rev_loss = cls_criterion(model_D(golden_rev_embedds), 1-labels.repeat(config.K))
    
    dis_loss = raw_loss + rec_loss + rev_loss + sample_raw_loss + sample_rev_loss
    
    optim_D.zero_grad()
    dis_loss.backward()
    torch.nn.utils.clip_grad_norm_(model_D.parameters(), config.max_grad_norm)
    optim_D.step()
    return dis_loss.item() / 5


def train(config, log, vocab, model_G, model_D, train_dataloader, dev_dataloader):
    
    optim_G = torch.optim.Adam(model_G.parameters(), lr=0.0, betas=(0.9, 0.999))
    optim_D = torch.optim.Adam(model_D.parameters(), lr=0.0, betas=(0.5, 0.999))

    optim_G = ISRScheduler(optimizer=optim_G, warmup_steps=config.warmup_steps,
        max_lr=config.max_lr, min_lr=config.min_lr, init_lr=config.init_lr, beta=0.75)

    optim_D = ISRScheduler(optimizer=optim_D, warmup_steps=config.warmup_steps,
            max_lr=config.max_lr, min_lr=config.min_lr, init_lr=config.init_lr, beta=0.5)

    seq_criterion = Criterion(vocab.pad)
    cls_criterion = Criterion(-1)

    global_steps = 0

    for epoch in range(config.epochs):
        for step, batch in enumerate(train_dataloader):

            if (global_steps) % config.update_step == 0:
                log.info("Updating query embeddings.. ")
                model_G.eval()
                model_G.update_retrieval()

            model_D.train()
            model_G.train()

            # train generator
            losses, rec_loss, cyc_loss, adv_loss, cos_loss, bow_loss = \
                    train_generator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion)
            
            # train disriminator
            for _ in range(config.d_step):
                dis_loss = train_disriminator(config, vocab, batch, model_G, model_D, optim_G, optim_D, seq_criterion, cls_criterion)
            
            if global_steps%20==0:
                log.info(
                    "epoch {}, step {}/{}, losses: {:.2f}, rec_loss: {:.2f}, cyc_loss: {:.2f}, adv_loss: {:.2f}, cos_loss: {:.2f}, bow_loss: {:.2f}, dis_loss: {:.2f}, lr_G: {:.5f}, lr_D:{:.5f}".format(
                    epoch+1, step+1, len(train_dataloader), losses, rec_loss, cyc_loss, adv_loss, cos_loss, bow_loss, dis_loss, optim_G.rate(), optim_D.rate()
                ))

            global_steps += 1            

        log.info("saving epoch {}...".format(epoch+1))
        torch.save(model_G, os.path.join(config.ckpts_dir, "G_{}.pt".format(epoch+1)))
        torch.save(model_D, os.path.join(config.ckpts_dir, "D_{}.pt".format(epoch+1)))
        
        results = []
        model_G.eval()
        with torch.no_grad():
            for loader in dev_dataloader.loaders:
                outputs = []
                for batch_test in loader:
                    outs, (_, _) = model_G(batch_test[0], 1 - batch_test[1])
                    outs = outs.argmax(dim=-1)
                    outs = convert_ids_to_tokens(outs, vocab)
                    outputs += outs
                results.append(outputs)

        for i, res in enumerate(results):
            with open(os.path.join(config.out_dir, "epoch_{}.{}".format(epoch+1, LABEL_MAP[config.dataset][i])), "w", encoding="utf8") as f:
                for line in res:
                    f.write(line + "\n")

    return