class Config(object):

    pretrain_model_dir = './pretrain_storage/xlm-roberta-base'
    model_storage_dir = './model_storage/retrieval_storage'
    checkpoint_path = './model_storage/retrieval_storage/finetuned_model.bin'
    name = 'best'

    seed = 2023
    total_epoches = 15
    batch_size = 4
    val_batch_size = 64 * 2
    accumulation_steps = 1
    lr = 5e-5
    warmup_ratio = 1e-1
    weight_decay = 1e-2
    eps = 1e-8
    clip_grad_norm = 1.0

    return_type = 'mean_pooling'  # mean_pooling, pooled_output, cls
    norm = False
    top_k = 100

    fp16 = False
    ema = False
    ema_decay = 0.999
    adv = False
    adv_eps = 1.0

    num_workers = 0
    prefetch = 16
    device = 'cuda'

    log_freq = 200

    query_sequence_length = 512
    context_sequence_length = 512

    valid_only = False

    debug = False
