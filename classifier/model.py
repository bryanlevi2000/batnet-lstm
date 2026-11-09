import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PulseLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, n_classes, dropout, attention_dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.attention_dropout = nn.Dropout(attention_dropout)

        # One output per class, each used as an independent sigmoid head.
        self.heads = nn.Linear(hidden_size, n_classes)

    def forward(self, x, lengths, return_steps=False):
        """x: (B, T, F) padded batch. lengths: int64 CPU tensor (B,)."""

        steps = x.size(1)

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=steps)
        h = self.dropout(self.norm(h))

        mask = torch.arange(steps, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        scores = self.attention(h).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=1))
        pooled_logits = self.heads((weights.unsqueeze(-1) * h).sum(dim=1))

        if return_steps:
            return pooled_logits, self.heads(h)

        return pooled_logits
