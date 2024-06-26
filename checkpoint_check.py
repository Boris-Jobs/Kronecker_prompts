import torch


def load_checkpoint(checkpoint_path, use_gpu=True):
    # 加载 checkpoint 文件到 GPU 或 CPU
    map_location = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def print_checkpoint_structure(checkpoint):
    # 打印 checkpoint 的键
    print("Checkpoint keys:", checkpoint.keys())

    # 获取 state_dict
    state_dict = checkpoint["state_dict"]

    # 打印每层的键和形状
    print("\nModel State Dict:")
    for key, value in state_dict.items():
        print(f"Layer: {key} | Shape: {value.shape}")


def analyze_attention_layers(state_dict):
    # 分析注意力层
    attention_layers = {k: v for k, v in state_dict.items() if "attn" in k}
    print("\nAttention Layers:")
    for key, value in attention_layers.items():
        print(f"Attention Layer: {key} | Shape: {value.shape}")


def main():
    checkpoint_path = "/scratch/project_2007023/boris/missing_aware_prompts/missing_aware_prompts/vilt/vilt_200k_mlm_itm.ckpt"  # 替换为你的ckpt路径
    checkpoint = load_checkpoint(checkpoint_path, use_gpu=True)
    print_checkpoint_structure(checkpoint)
    analyze_attention_layers(checkpoint["state_dict"])


if __name__ == "__main__":
    main()

"""
Checkpoint keys: dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'callbacks', 'optimizer_states', 'lr_schedulers', 'native_amp_scaling_state', 'state_dict', 'hparams_name', 'hyper_parameters'])

Model State Dict:
Layer: text_embeddings.position_ids | Shape: torch.Size([1, 40])  # NOTE 最大文本长度为40
Layer: text_embeddings.word_embeddings.weight | Shape: torch.Size([30522, 768])  # NOTE 词汇量为30522，每个词嵌入维度为768
Layer: text_embeddings.position_embeddings.weight | Shape: torch.Size([40, 768])  # NOTE 每个位置（40个）的嵌入维度为768
Layer: text_embeddings.token_type_embeddings.weight | Shape: torch.Size([2, 768])  # NOTE Token类型嵌入矩阵
Layer: text_embeddings.LayerNorm.weight | Shape: torch.Size([768])
Layer: text_embeddings.LayerNorm.bias | Shape: torch.Size([768])
Layer: token_type_embeddings.weight | Shape: torch.Size([2, 768])
Layer: transformer.cls_token | Shape: torch.Size([1, 1, 768])
Layer: transformer.pos_embed | Shape: torch.Size([1, 145, 768])  # NOTE 144个patch再加上1个分类token
Layer: transformer.patch_embed.proj.weight | Shape: torch.Size([768, 3, 32, 32])
Layer: transformer.patch_embed.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.0.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.0.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.0.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.0.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.0.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.0.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.0.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.0.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.0.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.0.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.0.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.0.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.1.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.1.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.1.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.1.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.1.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.1.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.1.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.1.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.1.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.1.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.1.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.1.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.2.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.2.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.2.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.2.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.2.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.2.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.2.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.2.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.2.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.2.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.2.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.2.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.3.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.3.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.3.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.3.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.3.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.3.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.3.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.3.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.3.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.3.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.3.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.3.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.4.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.4.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.4.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.4.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.4.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.4.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.4.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.4.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.4.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.4.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.4.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.4.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.5.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.5.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.5.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.5.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.5.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.5.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.5.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.5.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.5.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.5.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.5.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.5.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.6.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.6.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.6.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.6.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.6.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.6.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.6.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.6.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.6.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.6.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.6.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.6.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.7.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.7.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.7.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.7.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.7.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.7.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.7.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.7.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.7.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.7.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.7.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.7.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.8.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.8.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.8.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.8.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.8.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.8.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.8.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.8.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.8.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.8.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.8.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.8.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.9.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.9.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.9.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.9.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.9.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.9.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.9.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.9.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.9.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.9.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.9.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.9.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.10.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.10.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.10.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.10.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.10.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.10.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.10.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.10.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.10.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.10.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.10.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.10.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.11.norm1.weight | Shape: torch.Size([768])
Layer: transformer.blocks.11.norm1.bias | Shape: torch.Size([768])
Layer: transformer.blocks.11.attn.qkv.weight | Shape: torch.Size([2304, 768])
Layer: transformer.blocks.11.attn.qkv.bias | Shape: torch.Size([2304])
Layer: transformer.blocks.11.attn.proj.weight | Shape: torch.Size([768, 768])
Layer: transformer.blocks.11.attn.proj.bias | Shape: torch.Size([768])
Layer: transformer.blocks.11.norm2.weight | Shape: torch.Size([768])
Layer: transformer.blocks.11.norm2.bias | Shape: torch.Size([768])
Layer: transformer.blocks.11.mlp.fc1.weight | Shape: torch.Size([3072, 768])
Layer: transformer.blocks.11.mlp.fc1.bias | Shape: torch.Size([3072])
Layer: transformer.blocks.11.mlp.fc2.weight | Shape: torch.Size([768, 3072])
Layer: transformer.blocks.11.mlp.fc2.bias | Shape: torch.Size([768])
Layer: transformer.norm.weight | Shape: torch.Size([768])
Layer: transformer.norm.bias | Shape: torch.Size([768])
Layer: pooler.dense.weight | Shape: torch.Size([768, 768])
Layer: pooler.dense.bias | Shape: torch.Size([768])
Layer: mlm_score.bias | Shape: torch.Size([30522])
Layer: mlm_score.transform.dense.weight | Shape: torch.Size([768, 768])
Layer: mlm_score.transform.dense.bias | Shape: torch.Size([768])
Layer: mlm_score.transform.LayerNorm.weight | Shape: torch.Size([768])
Layer: mlm_score.transform.LayerNorm.bias | Shape: torch.Size([768])
Layer: mlm_score.decoder.weight | Shape: torch.Size([30522, 768])
Layer: itm_score.fc.weight | Shape: torch.Size([2, 768])
Layer: itm_score.fc.bias | Shape: torch.Size([2])

Attention Layers:
Attention Layer: transformer.blocks.0.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.0.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.0.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.0.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.1.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.1.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.1.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.1.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.2.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.2.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.2.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.2.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.3.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.3.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.3.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.3.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.4.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.4.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.4.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.4.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.5.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.5.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.5.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.5.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.6.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.6.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.6.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.6.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.7.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.7.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.7.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.7.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.8.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.8.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.8.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.8.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.9.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.9.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.9.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.9.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.10.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.10.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.10.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.10.attn.proj.bias | Shape: torch.Size([768])
Attention Layer: transformer.blocks.11.attn.qkv.weight | Shape: torch.Size([2304, 768])
Attention Layer: transformer.blocks.11.attn.qkv.bias | Shape: torch.Size([2304])
Attention Layer: transformer.blocks.11.attn.proj.weight | Shape: torch.Size([768, 768])
Attention Layer: transformer.blocks.11.attn.proj.bias | Shape: torch.Size([768])
"""
