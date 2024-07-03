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


######## basic settings ########
@ex.config
def config():
    num_gpus = 2
    num_nodes = 1
    exp_name = None
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 32

    test_ratio = 0.8
    test_type = None
    test_exp_name = None

    # fix backbone model (ViLT) weights
    fix_model = True

    # missing modality config
    missing_type = {"train": None, "val": None, "test": None}
    missing_ratio = {"train": test_ratio, "val": test_ratio, "test": test_ratio}
    both_ratio = 0.5  # missing both ratio
    missing_table_root = "./datasets/missing_tables/"
    simulate_missing = True
    with_delta_infer = None

    # missing_aware_prompts config
    prompt_type = None
    prompt_length = 16
    learnt_p = None
    prompt_layers = [0, 1, 2, 3, 4, 5]
    multi_layer_prompt = None

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
    learning_rate = 1e-2
    weight_decay = 0.02
    decay_power = 1
    max_epoch = 5
    warmup_steps = 100
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
    val_check_interval = None
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 16  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


######## dataset settings ########
@ex.named_config
def task_finetune_mmimdb():
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
    draw_false_image = 0
    val_check_interval = 0.2
    max_text_len = 1024


@ex.named_config
def task_finetune_hateful_memes():
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    draw_false_image = 0
    val_check_interval = 0.11
    max_text_len = 128


######## prompt settings ########
@ex.named_config
def kronecker_prompts():
    prompt_type = "kronecker"
    learnt_p = True
    multi_layer_prompt = True


@ex.named_config
def input_prompts():
    prompt_type = "input"
    learnt_p = True
    multi_layer_prompt = True


@ex.named_config
def none_prompts():
    prompt_type = "none"
    learnt_p = False
    multi_layer_prompt = False


######## missing settings ########
@ex.named_config
def trainm_i_testm_t():
    test_type = "text"
    missing_type = {"train": "image", "val": "image", "test": "text"}


@ex.named_config
def trainm_t_testm_t():
    test_type = "text"
    missing_type = {"train": "text", "val": "text", "test": "text"}


@ex.named_config
def trainm_b_testm_t():
    test_type = "text"
    missing_type = {"train": "both", "val": "both", "test": "text"}


@ex.named_config
def trainm_i_testm_i():
    test_type = "image"
    missing_type = {"train": "image", "val": "image", "test": "image"}


@ex.named_config
def trainm_t_testm_i():
    test_type = "image"
    missing_type = {"train": "text", "val": "text", "test": "image"}


@ex.named_config
def trainm_b_testm_i():
    test_type = "image"
    missing_type = {"train": "both", "val": "both", "test": "image"}


@ex.named_config
def trainm_i_testm_b():
    test_type = "both"
    missing_type = {"train": "image", "val": "image", "test": "both"}


@ex.named_config
def trainm_t_testm_b():
    test_type = "both"
    missing_type = {"train": "text", "val": "text", "test": "both"}


@ex.named_config
def trainm_b_testm_b():
    test_type = "both"
    missing_type = {"train": "both", "val": "both", "test": "both"}
