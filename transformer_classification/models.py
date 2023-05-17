# Models for transformer EEG classification
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, ff_dim, dropout, batch_first):
        super().__init__()

        self.vec_embedding = nn.Linear(input_dim, hid_dim, bias=False)
#         self.pos_embedding = nn.Embedding(1000, hid_dim)  # position embedding

        encoder_layer = nn.TransformerEncoderLayer(hid_dim, n_heads, ff_dim, dropout,
                                                   norm_first=True, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.fc = nn.Linear(hid_dim, output_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: [src_len, batch_size]
        # embed tokens and positions
        tok_embedded = self.dropout1(self.vec_embedding(src))
        pos_embedded = 0  # self.dropout2(self.pos_embedding(pos))  # [src_len, batch_size, hid_dim]
        embedded = tok_embedded + pos_embedded

        # encode sequence
        encoded = self.encoder(embedded)  # [src_len, batch_size, hid_dim]
        # get final output and apply linear layer
#         final_output = encoded.mean(dim=2)  # [batch_size, hid_dim]
        logits = self.fc(encoded)  # [batch_size, output_dim]

        return logits