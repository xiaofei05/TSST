import torch
import torch.nn as nn
import random
from elasticsearch import Elasticsearch
from .utils import convert_ids_to_tokens

elasticsearch = Elasticsearch("127.0.0.1:9000")

class SparseRetriever(nn.Module):

    def __init__(self, unpaired_sents, vocab, es_server, es_index, add_sos=True, add_eos=True):
        super().__init__()
        
        self.pad_id = vocab.pad
        self.eos_id = vocab.eos
        self.sos_id = vocab.sos
        self.vocab = vocab
        self.es_index = es_index

        self.unpaired_sents = unpaired_sents
        self.representations = None
        
        global elasticsearch
        elasticsearch = Elasticsearch(es_server)

    def build_data(self, sents, vocab, add_sos, add_eos):
        pass
        
    def update_representation(self, encoder, batch_size=32, device="cuda"):
        pass
    

    def process_batch(self, sents):
        max_len = max([len(sent) for sent in sents])
        batch = []
        for sent in sents:
            batch.append(sent + [self.pad_id] * (max_len - len(sent)))
        return torch.LongTensor(batch)
    
    def process_retrieve_outs(self, batch_samples, batch_style):
        batch_sample_tokens = []
        for i in range(len(batch_samples)):
            batch_sample_tokens.append([self.vocab.sos] + [self.vocab.word2id.get(w, self.vocab.unk) for w in batch_samples[i]] + [self.vocab.eos])
        
        batch_samples = self.process_batch(batch_sample_tokens).to(batch_style.device)
        # B, K, seq_length
        return batch_samples.view(batch_style.size(0), -1, batch_samples.size(-1))

    def get_query(self, sents, labels, K=10):
        # doc = {'query': {'match_all': {}}}
        req_head = {'index': self.es_index, 'type': '_doc'}
        request = []
        for i, sent in enumerate(sents):
            doc = {
                'query': {
                    'bool': {
                        'must_not': [
                            {"term": {"id": sent}}
                        ],
                        'must': [
                            {'match': {'content': sent}}
                        ],
                        'filter': [
                            {'term': {'label': labels[i]}}
                        ]
                    }
                },
                'from': 0,
                "size": K
            }
            request += [req_head, doc]
        
        _searched = elasticsearch.msearch(body=request)
        
        results = []
        for i, result in enumerate(_searched["responses"]):
            cur_result = []
            for hit in result['hits']['hits']:
                cur_result.append(hit["_source"]["content"].split())
            if len(cur_result) < K:
                cur_result += random.sample(self.unpaired_sents[labels[i]], K-len(cur_result))
            results += cur_result
        return results

    def retrieve(self, batch_inputs, batch_query, batch_style, topk=10):
        batch_sents = convert_ids_to_tokens(batch_inputs.cpu().tolist(), self.vocab)
        results = self.get_query(batch_sents, batch_style.cpu().tolist(), topk)
        return self.process_retrieve_outs(results, batch_style)