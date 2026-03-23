import argparse
import os
from types import SimpleNamespace

import librosa
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import Model


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def read_two_column_scp(scp_path):
    wav_items = []
    with open(scp_path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid scp format at line {line_num}: {line}. "
                    "Expected '<id> <path>'."
                )
            utt_id, wav_path = parts
            wav_items.append((utt_id, wav_path))
    return wav_items


class WavListDataset(Dataset):
    def __init__(self, wav_items, cut=64600):
        self.wav_items = wav_items
        self.cut = cut

    def __len__(self):
        return len(self.wav_items)

    def __getitem__(self, index):
        utt_id, wav_path = self.wav_items[index]
        wav, _ = librosa.load(wav_path, sr=16000)
        wav_pad = pad(wav, self.cut)
        return Tensor(wav_pad), utt_id


def build_model(checkpoint_path, ssl_cp_path, device):
    args = SimpleNamespace(ssl_cp_path=ssl_cp_path)
    model = Model(args, device).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Handle checkpoints saved from nn.DataParallel / DDP wrappers.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, data_loader, output_path, device, total_items):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    processed = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch_x, batch_paths in data_loader:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_scores = batch_out.data.cpu().numpy()

            for utt_id, score_pair in zip(batch_paths, batch_scores.tolist()):
                score_0 = score_pair[0]
                score_1 = score_pair[1]
                out_f.write(f"{utt_id} {score_0} {score_1}\n")
            processed += len(batch_paths)
            progress = (processed / total_items) * 100.0
            print(
                f"\rProgress: {processed}/{total_items} ({progress:.2f}%)",
                end="",
                flush=True,
            )
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference from a wav list and write scores."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--wav_list",
        type=str,
        required=True,
        help="Two-column scp file with '<id> <path>' per line.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output score file path.",
    )
    parser.add_argument(
        "--ssl_cp_path",
        type=str,
        default="xlsr2_300m.pt",
        help="Path to SSL pretrained checkpoint used by model.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cpu" or "cuda". Default: auto.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    wav_items = read_two_column_scp(args.wav_list)
    if len(wav_items) == 0:
        raise ValueError(f"No entries found in scp file: {args.wav_list}")

    dataset = WavListDataset(wav_items)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.checkpoint_path, args.ssl_cp_path, device)
    run_inference(model, data_loader, args.output_path, device, total_items=len(wav_items))
    print(f"Wrote {len(wav_items)} scores to {args.output_path}")


if __name__ == "__main__":
    main()
