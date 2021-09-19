
class YelpConfig:
    # model
    embedding_size = 256
    nlayers = 1
    bidirectional = True
    hidden_size = 512
    style_feature_size = 512
    num_filters = 64
    K = 5
    
    max_length = 18
    batch_size = 128

    # data
    dataset = "yelp"
    data_dir = './data/yelp/'
    out_dir = "./outputs/yelp/"
    ckpts_dir = "./ckpts/yelp/"

    retriever = "sparse"
    device = "cuda"
    # pretrain
    load_pretrain = True
    pretrain_epochs = 15

    # train
    d_step = 1
    warmup_steps = 3000
    max_lr = 8e-4
    min_lr = 5e-5
    init_lr = 1e-5

    epochs = 10
    max_grad_norm = 3.0

    update_step = 200
    teacher_forcing_ratio = 0.0
    attention_dropout = 0.1
    dropout = 0.1

    elasticsearch_server = "127.0.0.1:9200"
    elasticsearch_index = "yelp"


class GYAFCConfig:

    # model
    embedding_size = 256
    nlayers = 1
    bidirectional = True
    hidden_size = 512
    style_feature_size = 512
    num_filters = 64
    K = 5
    
    max_length = 32
    batch_size = 64

    # data
    dataset = "gyafc"

    data_dir = './data/gyafc/'
    out_dir = "./outputs/gyafc/"
    ckpts_dir = "./ckpts/gyafc/"

    retriever = "dense"

    device = "cuda"

    # pretrain
    load_pretrain = True
    pretrain_epochs = 15
    
    # train
    d_step = 5
    warmup_steps = 3000
    max_lr = 8e-4
    min_lr = 5e-5
    init_lr = 1e-5

    epochs = 15
    max_grad_norm = 3.0

    update_step = 200
    teacher_forcing_ratio = 0.0
    attention_dropout = 0.2
    dropout = 0.2

    elasticsearch_server = "127.0.0.1:9200"
    elasticsearch_index = "gyafc"

############# map ##########
LABEL_MAP = {
    "gyafc": ["informal", "formal"],
    "yelp": ["pos", "neg"],
}

CONFIG_MAP = {
    "yelp": YelpConfig,
    "gyafc": GYAFCConfig
}
