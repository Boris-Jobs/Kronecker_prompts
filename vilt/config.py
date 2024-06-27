from sacred import Experiment

ex = Experiment("ViLT", save_git_info=False)


def _loss_names(d):  # _ indicates it is not a part of public API.
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,
    }
    ret.update(d)  # input d as a dictionary to update ret.
    return ret


@ex.config  # every parameter decorated by @ex.config would be transferred into _config.
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 256  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None

    # fix backbone model (ViLT) weights
    fix_model = True

    # missing modality config
    missing_ratio = {"train": 0.1, "val": 0.1, "test": 0.1}
    missing_type = {
        "train": "both",
        "val": "both",
        "test": "both",
    }  # ['text', 'image', 'both'] in VL tasks
    both_ratio = 0.5  # missing both ratio
    missing_table_root = "./datasets/missing_tables/"
    simulate_missing = False

    # missing_aware_prompts config
    prompt_type = "kronecker"
    prompt_length = 16
    learnt_p = True  # learnable prompts?
    prompt_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = True

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
    max_text_len = 1024

@ex.named_config
def kronecker_prompts():
    prompt_type = "kronecker"
    learnt_p = True
    prompt_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = True

@ex.named_config
def input_prompts():
    prompt_type = "input"
    learnt_p = True
    prompt_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = True   

@ex.named_config
def none_prompts():
    prompt_type = "none"
    learnt_p = False
    multi_layer_prompt = False   

@ex.named_config
def task_finetune_hatememes():
    exp_name = "finetune_hatememes"
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.11
    weight_decay = 2e-2
    max_text_len = 128


@ex.named_config
def task_finetune_food101():
    exp_name = "finetune_food101"
    datasets = ["Food101"]
    loss_names = _loss_names({"food101": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
    max_text_len = 512


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/vilt/result"
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200



