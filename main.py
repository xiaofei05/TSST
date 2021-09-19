import os
from train import train
from data_loader import Vocab, get_dataloader
from model import TSST, Discriminator, DenseRetriever, SparseRetriever
from pretrain import pretrain
from logger import Log
from utils import read_file
from config import LABEL_MAP, CONFIG_MAP
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', type=str, default='yelp')
    args = parser.parse_args()

    config = CONFIG_MAP[args.dataset]()

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if not os.path.exists(config.ckpts_dir):
        os.makedirs(config.ckpts_dir)

    config.log_file = os.path.join(config.out_dir, "log.txt")

    log = Log(__name__, config.log_file).getlog()

    train_data = []
    test_data = []

    for label in LABEL_MAP[args.dataset]:
        cur_train_file = os.path.join(config.data_dir, "train.{}".format(label))
        cur_train_data = [sent.split() for sent in read_file(cur_train_file)]

        train_data.append(cur_train_data)
        
        cur_test_file = os.path.join(config.data_dir, "test.{}".format(label))
        cur_test_data = [sent.split() for sent in read_file(cur_test_file)]

        test_data.append(cur_test_data)

    vocab = Vocab(os.path.join(config.data_dir, "vocab.txt"))

    log.info(dict((name, getattr(config, name)) for name in dir(config) if not name.startswith("__")))
    
    pertrain_loader, train_loader = get_dataloader(
        train_data,
        vocab=vocab, 
        max_length=config.max_length, 
        batch_size=config.batch_size, 
        device=config.device,
        shuffle=True
    )

    pertest_loader, test_loader = get_dataloader(
        test_data,
        vocab=vocab, 
        max_length=config.max_length, 
        batch_size=config.batch_size, 
        device=config.device,
        shuffle=False
    )
    
    if config.retriever == "sparse":
        retrieval = SparseRetriever(train_data, vocab, config.elasticsearch_server, config.elasticsearch_index)
    else:
        retrieval = DenseRetriever(train_data, vocab)

    model_G = TSST(config, vocab, retrieval).to(config.device)
    model_D = Discriminator(config).to(config.device)

    pretrain(config, log, vocab, model_G, pertrain_loader, load_ckpt=config.load_pretrain)
    train(config, log, vocab, model_G, model_D, train_loader, test_loader)