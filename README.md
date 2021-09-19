# TSST
Code for EMNLP2021 paper “Transductive Learning for Unsupervised Text Style Transfer”.

## Requirements
* pytorch == 1.2.0
* kenlm
* transformers == 3.5.0
* nltk == 3.4.5
* Elasticsearch

Generated results (outputs) of our model are in the `./outputs/` directory.

## Dataset
First of all, you should rename the data file to the following format. Each line in the file is an original sentence.
* Yelp
  * The dataset can be downloaded [here](https://github.com/fastnlp/style-transformer/tree/master/data/yelp). You need to format the file names as `{train/dev/test/reference}.{pos/neg}`, and put them in the `./data/yelp/`.

* GYAFC
  * Please refer to the [repo](https://github.com/raosudha89/GYAFC-corpus). You need to format the file names as `{train/dev/test/reference}.{informal/formal}`, and put them in the `./data/gyafc/`. Multiple references are distinguished by adding a number at the end, like `reference.informal.0`, `reference.informal.1`, `reference.informal.2`, `reference.informal.3`.

## Usage

### Step 1: Preparing the data
```python
python preprocess.py --data_dir ./data/ --dataset [yelp/gyafc]
```

### Step 2: Train the model
The detailed training parameters of each dataset are in `config.py`.

```python
python main.py --dataset [yelp/gyafc]
```
The generated sentences of each epoch are saved in the `./outputs/[yelp/gyafc]/`.

### Retriever
* Dense retriever. modify the `config.retriever = dense`.
* Sparse retriever. modify the `config.retriever = sparse`, and set the elasticsearch server and index in `config.py`. Please note that the `get_query` in `model/SparseRetriever` should return the relevant sentences.

## Evaluation

You can download pretrained classifiers and language models (KenLM) [here](https://drive.google.com/file/d/1IUjDf90AlqrnNQB_uBA_3OO2IxJ4aDyE/view?usp=sharing), or train them by yourself as follows.

* Train the BERT classifiers using `evaluator/train_classifier.py`.
```python
python evaluator/train_classifier.py --dataset yelp --pretrained_dir_or_name bert-base-uncased  --max_len 18 --save_model evaluator/classifier/
# or
python evaluator/train_classifier.py --dataset gyafc --pretrained_dir_or_name bert-base-uncased --max_len 32 --save_model evaluator/classifier/
```

* Train the [KenLM](https://github.com/kpu/kenlm)
```sh
export kenlm=$YOUR_KENLM_DIR/build/bin/
chmod +x evaluator/train_lm.sh

bash evaluator/train_lm.sh yelp
# or
bash evaluator/train_lm.sh gyafc
```

* Evaluate the results
```
python evaluator/evaluator.py --dataset yelp --file outputs/yelp/epoch_10
```
It will report the automatic metrics of `outputs/yelp/epoch_10.pos` and `outputs/yelp/epoch_10.neg`

## Citation  
```
@inproceedings{xiao2021transductive,
    title = "Transductive Learning for Unsupervised Text Style Transfer",
    author = "Fei, Xiao  and
      Liang, Pang  and
      Yanyan, Lan  and
      Yan, Wang and
      Huawei Shen and
      Xueqi Cheng",
    booktitle = "Proceedings of the 2021 Conference on Empirical
Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## License
[MIT License](./LICENSE)