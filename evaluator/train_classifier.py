import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import BertTokenizer, BertForSequenceClassification

import sys
sys.path.append(".")
sys.path.append("..")
from config import LABEL_MAP

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(line.strip().lower())
    return data


def collate_fn(batch, pad_id, device):

    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    btc_size = len(batch)
    max_input_len = max([len(inputs) for inputs, label in batch])
    
    for btc_idx in range(btc_size):
        inputs, label = batch[btc_idx]
        input_len = len(inputs)
        input_ids.append(inputs + [pad_id] * (max_input_len - input_len))
        
        attention_mask.append([1] * input_len + [0] * (max_input_len - input_len))
        token_type_ids.append([0] * max_input_len)
        labels.append(label)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).to(device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long).to(device),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).to(device),
        "labels": torch.tensor(labels, dtype=torch.long).to(device),
    }

def get_dataloader(dataset, pad_id, device, batch_size, shuffle):
    cfn = lambda x: collate_fn(x, pad_id, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=cfn)

# Dataset
class TransferDataset(Dataset):
    def __init__(self, file_path_list, tokenizer, max_len):
        self.data = []
        self.labels = []
        for i, file_path in enumerate(file_path_list):
            sents = read_file(file_path)
            self.data += sents
            self.labels += [i] * len(sents)

        self.process_data(tokenizer, max_len)
    
    def process_data(self, tokenizer, max_len):
        print("Tokenizing sentences ...")
        for i in tqdm(range(len(self.data))):
            self.data[i] = tokenizer(self.data[i], max_length=max_len, truncation=True)["input_ids"]        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def evaluate(model, dev_dataloader):
    right_count = 0
    total_count = 0
    total_loss = []
    print("***** Running evaluating *****")
    model.eval()
    for step, batch in enumerate(dev_dataloader):
        with torch.no_grad():
            loss, predicted = model(**batch)
            pred_labels = torch.argmax(predicted, dim=-1)
            total_loss.append(loss.item())
            right_count += pred_labels.eq(batch["labels"]).sum().item()
            total_count += len(pred_labels)
    acc = right_count / total_count
    print('Evaluate acc: {:.4f}, loss: {:.6f}  '.format(acc, sum(total_loss)/len(total_loss)))
    return acc

def train(model, train_dataloader, dev_dataloader, args):
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataloader) * args.num_train_epochs
    )

    # Train
    print("***** Running training *****")

    global_step = 0
    best_result = 0

    model.to(args.device)
    for epoch in range(0, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):

            model.train()
            
            loss, _ = model(**batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            model.zero_grad()

            if step % 20 == 0:
                print('Train Epoch[{}] Step[{} / {}] - loss: {:.6f}  '.format(epoch+1, step+1, len(train_dataloader), loss.item()))  # , accuracy, corrects
            global_step += 1

            if (args.evaluate_step > 0 and global_step % args.evaluate_step == 0) or (epoch==int(args.num_train_epochs)-1 and step == len(train_dataloader)-1):                
                result = evaluate(model, dev_dataloader)
                print("best acc: %.4f, current acc: %.4f" % (best_result, result))
                if best_result <= result:
                    best_result = result
                    print("Saving model checkpoint to %s" % args.save_model)
                    model.save_pretrained(args.save_model)
                print()
    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="yelp")

    parser.add_argument('--pretrained_dir_or_name', type=str, default="../pretrained_models/bert/")

    parser.add_argument('--evaluate_step', type=int, default=300)
    parser.add_argument('--max_len', type=int, default=18)

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--dev_batch_size', type=int, default=32)
    parser.add_argument('--num_train_epochs', type=int, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=2e-5)

    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-8)

    parser.add_argument('--save_model', type=str, default='./evaluator/classifier/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # make dirs
    root = os.path.dirname(os.path.abspath("__file__"))
    args.save_model = os.path.join(root, os.path.join(args.save_model, args.dataset))
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)

    print(args)

    set_seed(42)
    labels = LABEL_MAP[args.dataset]

    # load pretrained model
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_dir_or_name)
    tokenizer.save_pretrained(args.save_model)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_dir_or_name, num_labels=len(labels))
    model = model.to(args.device)

    datasets = list()
    for mode in ["train", "dev", "test"]:
        mode_files_path_list = list()
        for label in labels:
            mode_files_path_list.append(os.path.join(root, "./data/{}/{}.{}".format(args.dataset, mode, label)))
        datasets.append(
            TransferDataset(mode_files_path_list, tokenizer, max_len=args.max_len)
        )

    pad_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    train_dataloader = get_dataloader(datasets[0], pad_id=pad_id, device=args.device, batch_size=args.train_batch_size, shuffle=True)
    
    dev_dataloader, test_dataloader = map(lambda x: get_dataloader(x, pad_id=pad_id, device=args.device, batch_size=args.dev_batch_size, shuffle=False), datasets[1:])
    train(model, train_dataloader, dev_dataloader, args)

    tokenizer = BertTokenizer.from_pretrained(args.save_model)
    model = BertForSequenceClassification.from_pretrained(args.save_model)
    model = model.to(args.device)
    evaluate(model, test_dataloader)