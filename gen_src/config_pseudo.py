class Config(object):

    fr_data_path = 'DAMO_ConvAI/FrDoc2BotGeneration'
    vi_data_path = 'DAMO_ConvAI/ViDoc2BotGeneration'
    ch_data_path = 'DAMO_ConvAI/ZhDoc2BotDialogue'
    en_data_path = 'DAMO_ConvAI/EnDoc2BotDialogue'

    pseudo_path = './results/test_pseudo.json'

    pretrain_model_dir = './gen_pretrain_model'
    model_storage_dir = './model_storage/gen_tmp_storage'
    checkpoint_path = './model_storage/gen_tmp_storage/finetuned_model_epoch15.bin'
    warmup_checkpoint_path = None
    name = 'best'

    seed = 2023
    total_epoches = 20
    batch_size = 4
    val_batch_size = 16
    accumulation_steps = 1
    encoder_lr = 1e-4
    decoder_lr = 2e-5
    other_lr = 2e-5
    opt_lr = 5e-5
    warmup_ratio = 1e-1
    weight_decay = 1e-2
    eps = 1e-8
    clip_grad_norm = 1.0

    loss_log_freq = 500

    num_beams = 1
    no_repeat_ngram_size = 20
    max_length = 512
    query_max_length = 128

    infer_num_beams = 1
    infer_no_repeat_ngram_size = 20
    infer_max_length = 512
    infer_query_max_length = 128

    fp16 = False
    ema = False
    ema_decay = 0.999
    adv = True
    adv_eps = 0.5
    awp_start = 15

    num_workers = 0
    prefetch = 16
    device = 'cuda'

    valid_only = False
    debug = False

    save_epoch =15

    inference_mode = 'pseudo'

    add_translate_data = False
    add_prompt = True
    add_pseudo = False

