import math
from typing import Tuple

import torch
from torch import nn, Tensor
from active_critic.model_src.base_transformer import DebugTEL, DebugTE

class ModelSetup:
    def __init__(self) -> None:
        self.d_output = 4
        self.nhead = 4
        self.d_hid = 512
        self.d_model = 512
        self.nlayers = 4
        self.seq_len = 100
        self.ntoken = -1
        self.dropout = 0.2
        self.device = 'cuda'

class TransformerModel(nn.Module):

    def __init__(self, model_setup:ModelSetup = None):
        super().__init__()
        self.model_type = 'Transformer'
        self.model_setup = model_setup
        self.lazy_init = False


    def _lazy_init(self, src: Tensor):
        ntoken = src.size(-1)
        d_output = self.model_setup.d_output
        d_model = self.model_setup.d_model
        nhead = self.model_setup.nhead
        d_hid = self.model_setup.d_hid
        nlayers = self.model_setup.nlayers
        dropout = self.model_setup.dropout
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = DebugTEL(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = DebugTE(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, d_model)
        self.decoder = nn.Linear(d_model, d_output)
        self.to(self.model_setup.device)
        self.lazy_init = True


    def forward(self, src: Tensor, mask = None, return_attention=False) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        if not self.lazy_init:
            self._lazy_init(src)
        src = self.encoder(src) * math.sqrt(self.model_setup.d_model)
        src = self.pos_encoder(src)
        if return_attention:
            output, attention = self.transformer_encoder.forward(src=src, mask=mask, return_attention=return_attention)
        else:
            output = self.transformer_encoder.forward(src=src, mask=mask)
        output = self.decoder(output)
        if return_attention:
            return output, attention
        else:
            return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
