"""INVESTIGATION ONLY: replay one epoch of a seeded run and dump details of gradient spikes.

Run from the project root:
    python classifier/replay_epoch.py --run "<run folder>" --epoch 39

Starts from <run>/epochs/epoch_<epoch-1>.pt and repeats the run's training steps with the
same seed, shuffle order, and dropout masks. Writes only to <run>/replay_epoch_<epoch>/.
The run's checkpoints and logs are not modified. Remove this file after the investigation
(see INVESTIGATION_CHANGES.md).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset import PulseDataset, collate_fn, load_split, read_manifest
from model import PulseLSTM
from train import seed_epoch, step_loss

GROUPS = ("lstm", "norm", "attention", "heads")
LABELS = {"lstm": "lstm", "norm": "layernorm", "attention": "attention", "heads": "heads"}


class Indexed(Dataset):
    """Same samples as PulseDataset, plus the dataset index. Does not change sampling."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, i


def collate_indexed(batch):
    x, lengths, labels = collate_fn([(s, y) for s, y, _ in batch])
    return x, lengths, labels, torch.tensor([i for _, _, i in batch])


def sample_sources(shard_dir, keep):
    """Map dataset index -> (wav_path, pulse_id) through metadata.jsonl."""

    positions = []
    for entry in read_manifest(shard_dir, "train"):
        shard = Path(entry["files"]["features"]).name.replace("_features.npy", "")
        positions += [(shard, i) for i in range(entry["num_samples"])]

    lookup = {}
    with open(shard_dir / "metadata.jsonl", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                m = json.loads(line)
                if m["split"] == "train":
                    lookup[(m["shard"], m["sample_index"])] = (m["wav_path"], m["pulse_id"])

    return [lookup.get(positions[k], ("?", -1)) for k in keep]


def group_norms(grads, names):
    sq = dict.fromkeys(GROUPS, 0.0)
    for name, g in zip(names, grads):
        if g is not None:
            sq[name.split(".")[0]] += g.float().pow(2).sum().item()
    return {k: v ** 0.5 for k, v in sq.items()}


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run folder of a seeded run")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch to replay (needs epoch-1 checkpoint)")
    parser.add_argument("--stop-step", type=int, default=None, help="Stop after this global step")
    parser.add_argument("--spike-threshold", type=float, default=1.0, help="Grad norm that counts as a spike")
    args = parser.parse_args()

    if config.SEED is None:
        raise RuntimeError("Replay needs SEED set to the value used by the run.")

    run_dir = Path(args.run)
    out_dir = run_dir / f"replay_epoch_{args.epoch:03d}"
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    shard_dir = Path(config.TRAIN_SHARD_DIR)
    with open(shard_dir / "class_mapping.json", encoding="utf-8") as file:
        class_mapping = json.load(file)
    names_by_label = sorted(class_mapping, key=class_mapping.get)
    n_classes = len(class_mapping)
    alpha = config.TARGET_REPLICATION_ALPHA

    features, offsets, labels, keep = load_split(
        shard_dir, "train", config.FEATURE_COLUMNS, config.MAX_SEQUENCE_LENGTH
    )
    data = PulseDataset(features, offsets, labels)
    sources = sample_sources(shard_dir, keep)

    state = torch.load(run_dir / "epochs" / f"epoch_{args.epoch - 1:03d}.pt", map_location=device)
    model = PulseLSTM(**state["model_args"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = config.LEARNING_RATE
        group["weight_decay"] = config.WEIGHT_DECAY

    param_names = [n for n, _ in model.named_parameters()]
    params = list(model.parameters())

    # Capture the LayerNorm input (LSTM output) and pre-dropout attention weights.
    captured = {}
    model.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("h", i[0].detach()))
    model.attention_dropout.register_forward_hook(lambda m, i, o: captured.__setitem__("w", i[0].detach()))

    generator = torch.Generator()
    loader = DataLoader(
        Indexed(data), batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_indexed,
        num_workers=0, pin_memory=True, generator=generator,
    )
    seed_epoch(config.SEED, args.epoch, generator)
    model.train()

    step = state["step"]
    steps_file = open(out_dir / "steps.csv", "w", newline="", encoding="utf-8")
    steps_writer = csv.writer(steps_file)
    steps_writer.writerow(["step", "loss", "grad_norm", *[f"grad_{LABELS[g]}" for g in GROUPS], "update_norm", "acc"])
    spike_rows = []
    print(f"replaying epoch {args.epoch} from step {step}, spike threshold {args.spike_threshold}")

    for x, lengths, batch_labels, idx in loader:

        step += 1
        x = x.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        y = F.one_hot(batch_labels, n_classes).float()

        # Identical to train.py (alpha > 0 branch).
        pooled_logits, step_logits = model(x, lengths, return_steps=True)
        loss = (
            alpha * step_loss(step_logits, y, lengths)
            + (1 - alpha) * F.binary_cross_entropy_with_logits(pooled_logits, y)
        )

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grads = [p.grad for p in params]
        grad_norm = torch.nn.utils.get_total_norm([g for g in grads if g is not None]).item()
        gnorms = group_norms(grads, param_names)

        if grad_norm > args.spike_threshold:

            # Per-sample terms from the same forward pass (same dropout masks).
            B = len(batch_labels)
            per_step = F.binary_cross_entropy_with_logits(
                step_logits, y[:, None, :].expand_as(step_logits), reduction="none"
            ).mean(dim=2)
            mask = torch.arange(per_step.size(1), device=device)[None, :] < lengths.to(device)[:, None]
            step_terms = per_step.masked_fill(~mask, 0).sum(dim=1) / lengths.to(device)
            pooled_terms = F.binary_cross_entropy_with_logits(pooled_logits, y, reduction="none").mean(dim=1)
            terms = (alpha * step_terms + (1 - alpha) * pooled_terms) / B

            h, w = captured["h"], captured["w"]
            probs = torch.sigmoid(pooled_logits.detach())

            for b in range(B):
                g = torch.autograd.grad(terms[b], params, retain_graph=True, allow_unused=True)
                gn = group_norms(g, param_names)
                L = int(lengths[b])
                hb = h[b, :L].float()
                std_per_step = hb.std(dim=1, unbiased=False)
                xb = x[b, :L].float()
                flat = (xb[1:] == xb[:-1]).all(dim=1).float().mean().item() if L > 1 else 0.0
                wav, pulse = sources[int(idx[b])]
                spike_rows.append({
                    "step": step, "batch_grad_norm": grad_norm,
                    "sample_grad_norm": sum(v ** 2 for v in gn.values()) ** 0.5,
                    **{f"sample_grad_{LABELS[k]}": v for k, v in gn.items()},
                    "dataset_index": int(idx[b]), "wav_path": wav, "pulse_id": pulse,
                    "label": names_by_label[int(batch_labels[b])],
                    "pred": names_by_label[int(pooled_logits[b].argmax())],
                    "true_prob": probs[b, batch_labels[b]].item(),
                    "length": L,
                    "pooled_loss": pooled_terms[b].item(), "step_loss": step_terms[b].item(),
                    "h_std_min": std_per_step.min().item(),
                    "h_std_min_at": int(std_per_step.argmin()),
                    "h_abs_max": hb.abs().max().item(),
                    "attn_max_weight": w[b, :L].max().item(),
                    "freq_min": xb[:, 0].min().item(), "freq_max": xb[:, 0].max().item(),
                    "amp_min": xb[:, 1].min().item(), "amp_max": xb[:, 1].max().item(),
                    "flat_step_frac": flat,
                })

            torch.save({"x": x.cpu(), "lengths": lengths, "labels": batch_labels.cpu(), "idx": idx,
                        "grads": {n: g.cpu() for n, g in zip(param_names, grads) if g is not None}},
                       out_dir / f"spike_batch_{step}.pt")
            print(f"  spike at step {step}: grad norm {grad_norm:.3g} "
                  + " ".join(f"{LABELS[k]} {v:.3g}" for k, v in gnorms.items()))

        before = torch.cat([p.detach().flatten() for p in params])
        optimizer.step()
        update_norm = (torch.cat([p.detach().flatten() for p in params]) - before).norm().item()
        acc = (pooled_logits.argmax(dim=1) == batch_labels).float().mean().item()

        steps_writer.writerow([step, f"{loss.item():.6f}", f"{grad_norm:.5g}",
                               *(f"{gnorms[k]:.5g}" for k in GROUPS), f"{update_norm:.5g}", f"{acc:.4f}"])

        if args.stop_step and step >= args.stop_step:
            break

    steps_file.close()
    pd.DataFrame(spike_rows).to_csv(out_dir / "spike_samples.csv", index=False)

    # Check the replay against the original run's step_diagnostics.csv.
    replay = pd.read_csv(out_dir / "steps.csv")
    replay["block"] = (replay.step - 1) // config.DIAG_LOG_EVERY
    blocks = replay.groupby("block").agg(step=("step", "max"), loss_mean=("loss", "mean"),
                                         grad_norm_max=("grad_norm", "max"))
    blocks = blocks[blocks.step % config.DIAG_LOG_EVERY == 0]
    original = pd.read_csv(run_dir / "step_diagnostics.csv").set_index("step")
    joined = blocks.set_index("step").join(original[["loss_mean", "grad_norm_max"]], rsuffix="_run")
    loss_diff = (joined.loss_mean - joined.loss_mean_run).abs().max()
    gn_diff = ((joined.grad_norm_max - joined.grad_norm_max_run).abs() / joined.grad_norm_max_run).max()
    print(f"replay vs run over {len(joined)} logged intervals: max loss_mean diff {loss_diff:.2e}, "
          f"max relative grad_norm_max diff {gn_diff:.2e}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
