# model.py
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config


class LSTMClassifier(nn.Module):
    """
    Sequence classifier using:
    Input Sequence  
    LSTM  
    LayerNorm  
    Attention Pooling   
    hidden-dimensional context vector 
    Independent one-vs-rest classifier heads 
    Class logits

    Each classifier head independently answers:

        "How strongly does this representation indicate class X?"

    There is one binary classifier head per class.
    """

    def __init__(self, num_classes):

        super().__init__()

        # --------------------------------------------------------------
        # LSTM
        # --------------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=len(config.FEATURE_COLUMNS),
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )

        # --------------------------------------------------------------
        # Layer normalization
        # --------------------------------------------------------------

        self.layernorm = nn.LayerNorm(
            config.HIDDEN_SIZE
        )

        # --------------------------------------------------------------
        # Attention
        # --------------------------------------------------------------

        self.attention = nn.Sequential(
            nn.Linear(
                config.HIDDEN_SIZE,
                config.HIDDEN_SIZE
            ),
            nn.Tanh(),
            nn.Dropout(
                config.ATTENTION_DROPOUT
            ),
            nn.Linear(
                config.HIDDEN_SIZE,
                1
            )
        )

        # --------------------------------------------------------------
        # Independent one-vs-rest classifier heads
        # --------------------------------------------------------------
        #
        # One independent binary classifier per class.
        #
        # Each head:
        #
        #     hidden-dimensional representation
        #         Linear(hidden representation to 1)
        #       binary class logit
        #
        # The heads do NOT share classifier weights.
        # They DO share the LSTM and attention representation.
        #

        self.classifiers = nn.ModuleList(
            [
                nn.Linear(
                    config.HIDDEN_SIZE,
                    1
                )
                for _ in range(num_classes)
            ]
        )

        self.num_classes = num_classes

    def _classify(self, representation):
        """
        Apply all independent classifier heads.

        Parameters
        ----------
        representation : torch.Tensor
            Shape:
                (batch_size, hidden_size)

        Returns
        -------
        logits : torch.Tensor
            Shape:
                (batch_size, num_classes)

        Each column corresponds to one independent binary
        one-vs-rest classifier.
        """

        logits = torch.cat(
            [
                classifier(representation)
                for classifier in self.classifiers
            ],
            dim=1
        )

        return logits

    def forward(self, x, lengths):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (batch_size, max_seq_len, num_features)

        lengths : torch.Tensor
            Shape:
                (batch_size,)

            True sequence lengths before padding.

        Returns
        -------
        step_logits : torch.Tensor
            Shape:
                (batch_size, max_seq_len, num_classes)

            Independent one-vs-rest logits for every valid timestep.

        final_logits : torch.Tensor
            Shape:
                (batch_size, num_classes)

            Independent one-vs-rest logits based on the
            attention-pooled sequence representation.

        Notes
        -----
        These are raw logits. No sigmoid is applied here.

        For inference:

            probabilities = torch.sigmoid(final_logits)

        should be used.
        """

        # --------------------------------------------------------------
        # Pack sequence
        # --------------------------------------------------------------

        packed_x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # --------------------------------------------------------------
        # LSTM
        # --------------------------------------------------------------

        packed_output, (hidden, cell) = self.lstm(
            packed_x
        )

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        # --------------------------------------------------------------
        # Layer normalization
        # --------------------------------------------------------------

        output = self.layernorm(
            output
        )

        # output shape:
        #
        # (batch_size, max_seq_len, hidden_size)

        # --------------------------------------------------------------
        # Step predictions
        #
        # Target replication
        # --------------------------------------------------------------

        batch_size = output.size(0)
        max_len = output.size(1)
        hidden_size = output.size(2)

        # Flatten the timestep dimension so that the classifier
        # heads can operate on every timestep.
        #
        # (batch, max_len, hidden)
        #          ↓
        # (batch * max_len, hidden)

        flattened_output = output.reshape(
            batch_size * max_len,
            hidden_size
        )

        flattened_step_logits = self._classify(
            flattened_output
        )

        # Restore the timestep dimension.
        #
        # (batch * max_len, num_classes)
        #          ↓
        # (batch, max_len, num_classes)

        step_logits = flattened_step_logits.reshape(
            batch_size,
            max_len,
            self.num_classes
        )

        # --------------------------------------------------------------
        # Attention pooling
        # --------------------------------------------------------------

        # Compute attention scores.

        attention_scores = self.attention(
            output
        ).squeeze(-1)

        # Shape:
        #
        # (batch_size, max_seq_len)

        # --------------------------------------------------------------
        # Mask padded timesteps
        # --------------------------------------------------------------

        mask = (
            torch.arange(
                max_len,
                device=lengths.device
            ).unsqueeze(0)
            <
            lengths.unsqueeze(1)
        )

        attention_scores = attention_scores.masked_fill(
            ~mask,
            float("-inf")
        )

        # --------------------------------------------------------------
        # Normalize attention scores
        # --------------------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # --------------------------------------------------------------
        # Weighted average of hidden states
        # --------------------------------------------------------------

        context = torch.sum(
            output *
            attention_weights.unsqueeze(-1),
            dim=1
        )

        # context shape:
        #
        # (batch_size, hidden_size)

        # --------------------------------------------------------------
        # Final independent classifier predictions
        # --------------------------------------------------------------

        final_logits = self._classify(
            context
        )

        return step_logits, final_logits
    