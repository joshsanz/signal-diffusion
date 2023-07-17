# Models for transformer EEG classification
import torch
import torch.nn as nn
import torch.nn.functional as F


activation_map = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
}


class TransformerSequenceClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, ff_dim, dropout, batch_first):
        super().__init__()
        self.batch_first = batch_first
        self.vec_embedding = nn.Linear(input_dim, hid_dim, bias=False)
#         self.pos_embedding = nn.Embedding(1000, hid_dim)  # position embedding

        encoder_layer = nn.TransformerEncoderLayer(hid_dim, n_heads, ff_dim, dropout,
                                                   norm_first=True, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.fc = nn.Linear(hid_dim, output_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [x_len, batch_size]
        # embed tokens and positions
        tok_embedded = self.dropout1(self.vec_embedding(x))
        pos_embedded = 0  # self.dropout2(self.pos_embedding(pos))  # [x_len, batch_size, hid_dim]
        embedded = tok_embedded + pos_embedded

        # encode sequence
        encoded = self.encoder(embedded)  # [x_len, batch_size, hid_dim]
        # get final output and apply linear layer
#         final_output = encoded.mean(dim=2)  # [batch_size, hid_dim]
        logits = self.fc(encoded)  # [batch_size, output_dim]

        return logits


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, ff_dim, dropout, batch_first):
        super().__init__()
        self.batch_first = batch_first
        self.vec_embedding = nn.Linear(input_dim, hid_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(hid_dim, n_heads, ff_dim, dropout,
                                                   norm_first=True, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(hid_dim, output_dim)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        tok_embedded = self.dropout1(self.vec_embedding(x))
        encoded = self.encoder(tok_embedded)
        pos_embedded = 0
        embedded = tok_embedded + pos_embedded

        encoded = self.encoder(embedded)
        # Do global avg pooling on sequence axis
        pool_dim = 1 if self.batch_first else 0
        pooled = encoded.mean(dim=pool_dim)
        logits = self.fc(pooled)
        return logits

# old ff dims: f_dims=[1000, 500, 250], dropout=0.5,
class CNNClassifier(nn.Module):
    def __init__(self, in_channels, out_dim,
                 conv_ks=[(7, 3), (5,3), (3, 3)], conv_cs=[8, 16, 128],
                 conv_ss=[1, 1, 1], conv_ps=[(2, 1), (1, 1), (1, 1)],
                 pool_ks=[(4, 2), (4, 2), (2, 2)], pool_ss=[(4, 2), (2, 2), (2, 2)],
                 ff_dims=[500, 250, 125], dropout=0.5,
                 pooling="max", activation="gelu"):
        super().__init__()
        # Store architecture sizes & strides
        self.conv_kernels = conv_ks
        self.conv_channels = conv_cs
        self.conv_strides = conv_ss
        self.conv_pads = conv_ps
        self.pool_ks = pool_ks
        self.pool_ss = pool_ss
        self.hidden_layers = ff_dims
        # Pooling type
        if pooling == "max":
            self.pooling = nn.MaxPool2d
        else:
            self.pooling = nn.AvgPool2d

        self.activation_fn = activation_map[activation]
        # Build conv layers
        conv_chan_list = [in_channels] + self.conv_channels

        self.convs = nn.ModuleList()
        for i in range(len(conv_ks)):
            self.convs.append(nn.Conv2d(conv_chan_list[i], conv_chan_list[i + 1],
                                        self.conv_kernels[i], stride=self.conv_strides[i],
                                        padding=self.conv_pads[i]))
            self.convs.append(self.activation_fn())
            self.convs.append(self.pooling(self.pool_ks[i], self.pool_ss[i]))
        # Build linear layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Dropout(dropout))
        self.fc.append(nn.LazyLinear(self.hidden_layers[0]))
        for i in range(len(ff_dims) - 1):
            self.fc.append(self.activation_fn())
            self.fc.append(nn.Dropout(dropout))
            self.fc.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))        
        self.fc.append(nn.Linear(self.hidden_layers[-1], out_dim))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(-1)
        for layer in self.fc:
            x = layer(x)
        return x


class CNNClassifierLight(nn.Module):
    def __init__(self, in_channels, out_dim,
                 conv_ks=[(5,5), (3, 3)], conv_cs=[3, 8],
                 conv_ss=[1, 1], conv_ps=[(2, 1), (1, 1)],
                 pool_ks=[(4, 2), (2, 2)], pool_ss=[(4, 2), (2, 2)],
                 ff_dims=[250, 125], dropout=0.5,
                 pooling="max", activation="gelu"):
        super().__init__()
        # Store architecture sizes & strides
        self.conv_kernels = conv_ks
        self.conv_channels = conv_cs
        self.conv_strides = conv_ss
        self.conv_pads = conv_ps
        self.pool_ks = pool_ks
        self.pool_ss = pool_ss
        self.hidden_layers = ff_dims
        # Pooling type
        if pooling == "max":
            self.pooling = nn.MaxPool2d
        else:
            self.pooling = nn.AvgPool2d

        self.activation_fn = activation_map[activation]
        # Build conv layers
        conv_chan_list = [in_channels] + self.conv_channels

        self.convs = nn.ModuleList()
        for i in range(len(conv_ks)):
            self.convs.append(nn.Conv2d(conv_chan_list[i], conv_chan_list[i + 1],
                                        self.conv_kernels[i], stride=self.conv_strides[i],
                                        padding=self.conv_pads[i]))
            self.convs.append(self.activation_fn())
            self.convs.append(self.pooling(self.pool_ks[i], self.pool_ss[i]))
        # Build linear layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Dropout(dropout))
        self.fc.append(nn.LazyLinear(self.hidden_layers[0]))
        for i in range(len(ff_dims) - 1):
            self.fc.append(self.activation_fn())
            self.fc.append(nn.Dropout(dropout))
            self.fc.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))        
        self.fc.append(nn.Linear(self.hidden_layers[-1], out_dim))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(-1)
        for layer in self.fc:
            x = layer(x)
        return x
