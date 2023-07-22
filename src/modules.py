import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

"""
Summary:
1) X: (batch_size, seq_len, d)
2) W_Q:

Mask D-> upper triangular matrix

"""

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        intermediate_layer_dim,
        output_dim,
        num_intermediate_layers=1,
        activation="gelu",
    ) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, intermediate_layer_dim)]
            + [nn.Linear(intermediate_layer_dim, intermediate_layer_dim)]
            * (num_intermediate_layers - 1)
        )
        self.output_layer = nn.Linear(intermediate_layer_dim, output_dim)
        if activation == "gelu":
            self.activation = nn.GELU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class GatedMultiScaleRetentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(GatedMultiScaleRetentionLayer, self).__init__()
        self.gamma = 1 - torch.exp(
            (-5 - torch.arange(0, num_heads)) * torch.log(torch.tensor(2))
        )  # shape: [num_heads]

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim=32, use_xpos=True)

        self.W_g = nn.Linear(d_model, d_model)

        ## Isn't this same as Instance Norm?
        self.group_norm = nn.GroupNorm(num_heads, num_heads)
        self.swish = nn.SiLU()
        self.W_o = nn.Linear(d_model, d_model)
        self.norm_term = torch.sqrt(torch.tensor(d_model // num_heads))

    def forward(self, x):
        # x-> batch_size, seq_len, d_model
        batch_size, seq_len, d_model = x.shape
        q = self.transform_into_heads(self.W_q(x))
        k = self.transform_into_heads(self.W_k(x))

        q, k = self.rotary_embedding.rotate_queries_and_keys(q, k)
        
        v = self.transform_into_heads(self.W_v(x))
        retention_scores = self.group_norm(self.parallel_retention(q, k, v))
        concat_scores = torch.transpose(retention_scores, 1, 2).flatten(
            2, 3
        )  # batch_size, seq_len, d_model

        output = self.W_o(self.swish(self.W_g(x)) * concat_scores)
        return output

    def parallel_retention(self, q, k, v, decay_mask):
        retention = q @ k.transpose(-1, -2)
        retention = retention * decay_mask
        output = retention @ v
        output = self.group_norm(output)
        return output
        # batch_size, num_heads, seq_len, d = q.shape
        # masks = self.get_masks(seq_len)
        # retention_scores = (q @ k.transpose(-1, -2) / self.norm_term) * masks
        # retention_scores = torch.div(
        #     retention_scores, retention_scores.sum(-1).unsqueeze(-1)
        # )
        # return retention_scores @ v
        ## Now, q, k, v are of dim:  batch_size, num_heads, seq_len, d
    
    def recurrent_retention(self, q, k, v, past_kv, decay):
        current_kv = decay * past_kv + k.unsqueeze(-1) * v.unsqueeze(-2)
        output = torch.sum(q.unsqueeze(-1) * current_kv, dim=-2)
        output = torch.sum(q.unsqueeze(-1) * current_kv, dim=-2)
        output = self.group_norm(output)
        return output, current_kv

    def chunkwise_retention(self, q, k, v, past_kv, decay_mask, chunk_decay, inner_decay):
        retention = q @ k.transpose(-1, -2)
        retention = retention * decay_mask
        inner_retention = retention @ v
        cross_retention = (q @ past_kv) * inner_decay
        retention = inner_retention + cross_retention
        output = self.group_norm(retention)
        current_kv = chunk_decay * past_kv + k.transpose(-1, -2) @ v
        return output, current_kv

    def transform_into_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return torch.transpose(
            x.view(batch_size, seq_len, self.num_heads, d_model // self.num_heads), 1, 2
        )

    def get_masks(self, seq_len):
        row_indices = torch.arange(seq_len).view(-1, 1)
        column_indices = torch.arange(seq_len).view(1, -1)
        power_matrix = row_indices - column_indices
        gamma_expanded = self.gamma.view(-1, 1, 1)
        masks = torch.where(
            power_matrix >= 0, gamma_expanded**power_matrix, torch.zeros(1, 1)
        )
        norm_masks = torch.div(masks, torch.sqrt(masks.sum(-1).unsqueeze(-1)))
        return norm_masks


class RetentionNetworkBlock(nn.Module):
    def __init__(
        self, d_model, num_heads, ffn_intermediate_layer_dim, ffn_intermediate_layers=1
    ):
        super(RetentionNetworkBlock, self).__init__()
        assert d_model % num_heads == 0
        self.ffn = FeedForwardNetwork(
            input_dim=d_model,
            intermediate_layer_dim=ffn_intermediate_layer_dim,
            output_dim=d_model,
            num_intermediate_layers=ffn_intermediate_layers,
        )
        self.multi_scale_retention = GatedMultiScaleRetentionLayer(d_model, num_heads)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.multi_scale_retention(self.layer_norm_1(x)) + x
        output = self.ffn(self.layer_norm_2(y)) + y
        return output


class RetentionNetwork(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_blocks,
        ffn_intermediate_layer_dim,
        ffn_intermediate_layers=1,
    ):
        super(RetentionNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [
                RetentionNetworkBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_intermediate_layer_dim=ffn_intermediate_layer_dim,
                    ffn_intermediate_layers=ffn_intermediate_layers,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    retnet = RetentionNetwork(
        d_model=512,
        num_heads=8,
        num_blocks=6,
        ffn_intermediate_layer_dim=1024,
        ffn_intermediate_layers=1,
    )
    rand_input = torch.rand(8, 76, 512)
    output = retnet(rand_input)
    print(output.shape)
