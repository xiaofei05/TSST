import os
import argparse
from config import LABEL_MAP
import collections
from utils import read_file
from nltk.tokenize import word_tokenize

def process_file(file_dir):
    print("Processing ", file_dir)
    for file in os.listdir(file_dir):
        for mode in ["train.", "dev.", "test.", "reference."]:
            if mode not in file:
                continue
            file_path = os.path.join(file_dir, file)
            data = read_file(file_path)
            new_data = [" ".join(word_tokenize(d)) for d in data]
            with open(file_path, "w", encoding="utf8") as f:
                for nd in new_data:
                    f.write(nd+"\n")

def build_vocab(sents):
    # word frequency
    word_freq = collections.defaultdict(int)
    for sent in sents:
        for word in sent.split():
            word_freq[word] += 1
    
    return word_freq

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--min_word_freq', type=int, default=0)
    args = parser.parse_args()

    # format files
    data_fold = os.path.join(args.data_dir, args.dataset)
    process_file(data_fold)

    # build vocab 
    file_list = [os.path.join(data_fold, f"train.{label}") for label in LABEL_MAP[args.dataset]]
    save_words = []
    for file in file_list:
        print(f"Building vocab of {file} ... ")
        data = read_file(file)
        save_words.append(build_vocab(data))

    vocab = collections.defaultdict(list)
    for idx, save_word in enumerate(save_words):
        for word in save_word:
            vocab[word] = vocab.get(word, [0]*len(save_words))
            vocab[word][idx] = save_word[word]
    
    filted_vocab = dict(filter(lambda x: sum(x[1]) > args.min_word_freq, vocab.items()))

    vocab_file = os.path.join(data_fold, "vocab.txt")
    with open(vocab_file, 'w', encoding='utf8') as f:
        for token in filted_vocab:
            f.write("\t".join(map(str, [token] + filted_vocab[token])) + "\n")