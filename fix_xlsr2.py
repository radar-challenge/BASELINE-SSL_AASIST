import torch
import argparse
from omegaconf import DictConfig, open_dict
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description="Fix pretrain checkpoint config keys.")
parser.add_argument("input_pt", help="Path to input .pt checkpoint")
parser.add_argument("output_pt", help="Path to output .pt checkpoint")
args = parser.parse_args()

cp=torch.load(args.input_pt,weights_only=False)
cfg = DictConfig(cp["cfg"])

dd = OmegaConf.to_container(cfg, resolve=True)
for k,v in dd.items():
    if not isinstance(v, dict):
        continue
    for key, _ in v.items():
        if key == "multiple_train_files":
            print(k)
            break
with open_dict(cfg):
    cfg.task.pop("multiple_train_files")
    cfg.task.pop("eval_wer")
    cfg.task.pop("eval_wer_config")
    cfg.task.pop("eval_wer_tokenizer")
    cfg.task.pop("eval_wer_post_process")
    cfg.task.pop("autoregressive")
    cfg.task.pop("shuffle_by_bucket")
    cfg.task.pop("shuffle_by_bucket_size")
    cfg.task.pop("mask_length")
    cfg.task.pop("mask_prob")
    cfg.task.pop("mask_selection")
    cfg.task.pop("mask_other")
    cfg.task.pop("no_mask_overlap")
    cfg.task.pop("mask_min_space")
    cfg.task.pop("mask_channel_length")
    cfg.task.pop("mask_channel_prob")
    cfg.task.pop("mask_channel_selection")
    cfg.task.pop("mask_channel_other")
    cfg.task.pop("no_mask_channel_overlap")
    cfg.task.pop("mask_channel_min_space")
    cfg.task.pop("conv_feature_layers")
    cfg.task.pop("encoder_embed_dim")
    cfg.task.pop("train_subset")

cp['cfg'] = OmegaConf.to_container(cfg, resolve=True)
torch.save(cp, args.output_pt)
