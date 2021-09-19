import os
import math
import kenlm
import torch
import argparse
from nltk.translate.bleu_score import corpus_bleu
from transformers import BertTokenizer, BertForSequenceClassification

import sys
sys.path.append(".")
sys.path.append("..")
from config import LABEL_MAP
from utils import read_file

class Evaluator:
    def __init__(self, dataset_name="yelp", device="cpu"):
        
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.join(os.path.dirname(self.cur_dir), "data"), dataset_name)

        self.dataset_name = dataset_name
        
        # original data and references for self-bleu and ref-bleu
        self.ori_data = self.__get_data__(file_name="test")
        self.ref_data = self.__get_data__(file_name="reference")

        # classifier for acc
        classifier_path = os.path.join(os.path.join(self.cur_dir, "classifier"), "{}".format(dataset_name))
        self.device = device
        self.bert = BertForSequenceClassification.from_pretrained(classifier_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(classifier_path)
        
        # language model for ppl
        self.lm_model = []
        for label in LABEL_MAP[self.dataset_name]:
            lm_path = os.path.join(os.path.join(self.cur_dir, "kenlm"), "{}.{}.bin".format(dataset_name, label))
            self.lm_model.append(kenlm.LanguageModel(lm_path))

    def __get_data__(self, file_name="test"):
        data = []
        for label in LABEL_MAP[self.dataset_name]:
            if self.dataset_name == "gyafc" and file_name == "reference":
                gyafc_ref = [read_file(os.path.join(self.data_dir, "{}.{}.{}".format(file_name, label, i))) for i in range(4)]
                label_data = list(zip(gyafc_ref[0], gyafc_ref[1], gyafc_ref[2], gyafc_ref[3]))
                data += [[sent.split() for sent in sents] for sents in label_data]
            else:
                label_data = read_file(os.path.join(self.data_dir, "{}.{}".format(file_name, label)))
                data += [[sent.split()] for sent in label_data]
        return data

    def get_ref_bleu(self, seg_sents):
        try:
            assert len(seg_sents) == len(self.ref_data)
        except:
            print(len(seg_sents))
        return corpus_bleu(self.ref_data, seg_sents) * 100

    def get_self_bleu(self, seg_sents):
        try:
            assert len(seg_sents) == len(self.ori_data)
        except:
            print(len(seg_sents))
        return corpus_bleu(self.ori_data, seg_sents) * 100

    def get_ppl(self, sents, labels):
        total_score = 0
        total_length = 0
        for sent, label in zip(sents, labels):
            total_score += self.lm_model[1-label].score(sent)
            total_length += len(sent.split())
        
        if total_length == 0:
            print(total_score, total_length)
            return math.pow(10, 4)
        else:
            return math.pow(10, -total_score/(total_length))

    def get_acc(self, sents, labels):
        batch = 32
        right_count = 0
        total_count = 0
        for i in range(0, len(sents), batch):
            predict = self.bert(**(self.tokenizer(sents[i:i+batch], max_length=256, padding=True, return_tensors="pt").to(self.device)))[0]
            preds = torch.argmax(predict, dim=-1).cpu()
            l = torch.LongTensor(labels[i:i+batch])
            right_count += preds.ne(l).sum()
            total_count += preds.size(0)
        
        assert len(sents) == total_count
        return right_count.item()/total_count * 100

    def evaluate(self, transfered_sents, labels):
        
        acc = self.get_acc(transfered_sents, labels)
        ppl = self.get_ppl(transfered_sents, labels)
        
        seg_sents = [sent.split() for sent in transfered_sents]
        self_bleu = self.get_self_bleu(seg_sents)
        ref_bleu = self.get_ref_bleu(seg_sents)

        gm = math.pow(acc * self_bleu * ref_bleu * 1.0 / math.log(ppl), 1.0/4.0)
    
        eval_str = "ACC: {:.1f} \tself-BLEU: {:.2f} \tref-BLEU: {:.2f} \tPPL: {:.0f} \tGM: {:.2f}".format(acc, self_bleu, ref_bleu, ppl, gm)
        return eval_str, (acc, self_bleu, ref_bleu, ppl, gm)

    def evaluate_file(self, result_file):
        transfered_sents = []
        labels = []
        for i, label in enumerate(LABEL_MAP[self.dataset_name]):
            sents = read_file("{}.{}".format(result_file, label))
            transfered_sents += sents
            labels += [i] * len(sents)
        
        eval_str, (acc, self_bleu, ref_bleu, ppl, gm) = self.evaluate(transfered_sents, labels)
        return eval_str, (acc, self_bleu, ref_bleu, ppl, gm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--datadir', type=str, default="outputs")
    parser.add_argument('--file', type=str, default="all")
    parser.add_argument('--cpu', action='store', default=False)
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Evaluating dataset: {dataset_name}")

    if args.file == "all":
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(os.path.join(os.path.dirname(cur_dir), args.datadir), dataset_name)
        result_files = set()
        for file in os.listdir(outputs_dir):
            file_split = file.split(".")
            if len(file_split) == 2 and file_split[0] != "log":
                result_files.add(os.path.join(outputs_dir, file_split[0]))
        result_files = sorted(list(result_files))
    else:
        result_files = [args.file]

    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(dataset_name, device=device)
    for result_file in result_files:
        print(result_file, end="\t")
        eval_str, (acc, self_bleu, ref_bleu, ppl, gm) = evaluator.evaluate_file(result_file)
        print(eval_str)