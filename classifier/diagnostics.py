"""INVESTIGATION ONLY: training diagnostics for the training-collapse bug.

Remove this file when the investigation is done (see INVESTIGATION_CHANGES.md).
"""

import csv

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


GROUPS = ("lstm", "norm", "attention", "heads")

STEP_COLUMNS = [
    "epoch", "step", "lr",
    # Averages and maxima over the steps since the previous row.
    "loss_mean", "loss_max", "acc", "grad_norm_mean", "grad_norm_max",
    "clipped_frac", "nonfinite_steps",
    # Values at the logged step only.
    "grad_lstm", "grad_layernorm", "grad_attention", "grad_heads",
    "param_norm", "update_ratio", "update_max_abs",
    "logit_abs_max", "top_logit_median",
    "attn_max_weight_median", "attn_max_weight_p99", "attn_eff_steps_frac_median",
]


@torch.no_grad()
def attention_stats(model, x, lengths):
    """Same math as PulseLSTM.forward in eval mode, returning attention summary values."""

    was_training = model.training
    model.eval()

    steps = x.size(1)
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    h, _ = model.lstm(packed)
    h, _ = pad_packed_sequence(h, batch_first=True, total_length=steps)
    h = model.norm(h)
    mask = torch.arange(steps, device=x.device)[None, :] < lengths.to(x.device)[:, None]
    scores = model.attention(h).squeeze(-1).masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=1)

    model.train(was_training)

    max_w = w.max(dim=1).values
    entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
    eff_frac = entropy.exp() / lengths.to(x.device).float()

    return max_w.median().item(), torch.quantile(max_w, 0.99).item(), eff_frac.median().item()


class Diagnostics:
    """Writes step_diagnostics.csv (every `every` optimizer steps) and epoch_diagnostics.csv."""

    def __init__(self, run_dir, every, clip_norm, class_names, start_step=0):

        self.every = every
        self.clip_norm = clip_norm
        self.class_names = class_names
        self.step = start_step
        self.lr = 0.0  # set by train.py at the start of every epoch

        self.step_file = open(run_dir / "step_diagnostics.csv", "w", newline="", encoding="utf-8")
        self.step_writer = csv.writer(self.step_file)
        self.step_writer.writerow(STEP_COLUMNS)

        self.epoch_file = open(run_dir / "epoch_diagnostics.csv", "w", newline="", encoding="utf-8")
        self.epoch_writer = csv.writer(self.epoch_file)
        self.epoch_writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds", "end_step", "lr"]
            + [f"val_acc_{name}" for name in class_names]
        )

        self._reset()

    def _reset(self):
        self.losses, self.norms, self.correct, self.count = [], [], [], 0

    def start_step(self):
        """Advance the global step. Returns True when this step gets a full row."""
        self.step += 1
        return self.step % self.every == 0

    def before_update(self, model):
        """Call after backward and clipping, before optimizer.step(), on log steps only."""

        sq = dict.fromkeys(GROUPS, 0.0)
        for name, p in model.named_parameters():
            if p.grad is not None:
                sq[name.split(".")[0]] += p.grad.float().pow(2).sum().item()

        self.group_norms = {k: v ** 0.5 for k, v in sq.items()}
        self.before = torch.cat([p.detach().flatten() for p in model.parameters()])

    def after_update(self, model, epoch, loss, grad_norm, logits, labels, x, lengths, log_now):

        self.losses.append(loss.detach())
        self.norms.append(grad_norm.detach())
        self.correct.append((logits.argmax(dim=1) == labels).sum())
        self.count += len(labels)

        if not log_now:
            return

        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        delta = after - self.before
        param_norm = after.norm().item()

        losses = torch.stack(self.losses).float()
        norms = torch.stack(self.norms).float()
        finite = torch.isfinite(norms)
        clipped = (norms > self.clip_norm).float().mean().item() if self.clip_norm else 0.0
        attn = attention_stats(model, x, lengths)
        logits = logits.detach().float()

        self.step_writer.writerow([
            epoch, self.step, f"{self.lr:.4g}",
            f"{losses.mean().item():.6f}", f"{losses.max().item():.6f}",
            f"{100 * torch.stack(self.correct).sum().item() / self.count:.2f}",
            f"{norms[finite].mean().item() if finite.any() else float('nan'):.5g}",
            f"{norms[finite].max().item() if finite.any() else float('nan'):.5g}",
            f"{clipped:.4f}", int((~finite).sum().item()),
            *(f"{self.group_norms[k]:.5g}" for k in GROUPS),
            f"{param_norm:.6g}", f"{delta.norm().item() / param_norm:.5g}",
            f"{delta.abs().max().item():.5g}",
            f"{logits.abs().max().item():.4g}", f"{logits.max(dim=1).values.median().item():.4g}",
            *(f"{v:.4g}" for v in attn),
        ])
        self.step_file.flush()
        self._reset()

    def write_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, seconds, class_counts):

        correct, total = class_counts[:, 0], class_counts[:, 1].clamp_min(1)
        per_class = (100 * correct / total).tolist()

        self.epoch_writer.writerow(
            [epoch, f"{train_loss:.6f}", f"{train_acc:.2f}", f"{val_loss:.6f}", f"{val_acc:.2f}",
             f"{seconds:.0f}", self.step, f"{self.lr:.4g}"]
            + [f"{v:.2f}" for v in per_class]
        )
        self.epoch_file.flush()

    def close(self):
        self.step_file.close()
        self.epoch_file.close()
