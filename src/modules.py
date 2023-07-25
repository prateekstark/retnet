import torch
import torch.nn as nn
import torch.nn.functional as F
from base import FeedForwardNetwork, GatedMultiScaleRetentionLayer


class RetentiveNetworkBlock(nn.Module):
    def __init__(
        self,
        d_model,
        k_dim,
        v_dim,
        num_heads,
        ffn_intermediate_layer_dim,
        ffn_intermediate_layers=1,
    ):
        super(RetentiveNetworkBlock, self).__init__()
        assert d_model % num_heads == 0
        self.ffn = FeedForwardNetwork(
            input_dim=d_model,
            intermediate_layer_dim=ffn_intermediate_layer_dim,
            output_dim=d_model,
            num_intermediate_layers=ffn_intermediate_layers,
        )
        self.multi_scale_retention = GatedMultiScaleRetentionLayer(
            d_model, k_dim, v_dim, num_heads
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, paradigm, past_kv=None, chunk_index=0):
        assert paradigm in ["parallel", "recurrent", "chunkwise"]

        if paradigm == "recurrent" and len(x.shape) == 2:
            x.unsqueeze_(dim=1)

        retention_scores, current_kv = self.multi_scale_retention(
            self.layer_norm_1(x),
            past_kv,
            paradigm,
            chunk_index=chunk_index,
        )

        y = retention_scores + x
        output = self.ffn(x) + y

        # output = self.ffn(self.layer_norm_2(y)) + y
        return output, current_kv


class RetentiveNetwork(nn.Module):
    def __init__(
        self,
        d_model,
        k_dim,
        v_dim,
        num_heads,
        num_blocks,
        ffn_intermediate_layer_dim,
        ffn_intermediate_layers=1,
    ):
        super(RetentiveNetwork, self).__init__()
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList(
            [
                RetentiveNetworkBlock(
                    d_model=d_model,
                    k_dim=k_dim,
                    v_dim=v_dim,
                    num_heads=num_heads,
                    ffn_intermediate_layer_dim=ffn_intermediate_layer_dim,
                    ffn_intermediate_layers=ffn_intermediate_layers,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, paradigm, chunk_index=0):
        past_kvs = [None] * self.num_blocks
        assert paradigm in ["parallel", "recurrent", "chunkwise"]

        if paradigm == "recurrent" and len(x.shape) == 2:
            x.unsqueeze_(dim=1)

        for i, layer in enumerate(self.layers):
            x, current_kvs = layer(x, past_kvs[i], paradigm, chunk_index=chunk_index)
            past_kvs[i] = current_kvs
        return x
