import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from xpos_relative_position import XPOS
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
    def __init__(self, d_model, k_dim, v_dim, num_heads):
        super(GatedMultiScaleRetentionLayer, self).__init__()
        self.gamma = 1 - torch.exp(
            (-5 - torch.arange(0, num_heads)) * torch.log(torch.tensor(2))
        )  # shape: [num_heads]

        self.W_q = nn.Linear(d_model, k_dim * num_heads)
        self.W_k = nn.Linear(d_model, k_dim * num_heads)
        self.W_v = nn.Linear(d_model, v_dim * num_heads)
        self.num_heads = num_heads
        self.rotary_embedding = RotaryEmbedding(dim=(k_dim * num_heads), use_xpos=True)
        self.xpos = XPOS(32, scale_base=512)
        self.W_g = nn.Linear(d_model, v_dim * num_heads)

        ## Isn't this same as Instance Norm?
        self.group_norm_concat = nn.GroupNorm(
            num_groups=num_heads, num_channels=d_model, affine=False
        )
        self.swish = nn.SiLU()
        self.W_o = nn.Linear(v_dim * num_heads, d_model)
        self.norm_term = torch.sqrt(torch.tensor(d_model // num_heads))
        self.k_dim = k_dim

    def forward(self, x, past_kv=None, paradigm="parallel", chunk_index=0):
        # x-> batch_size, seq_len, d_model
        assert paradigm in ["parallel", "recurrent", "chunkwise"]

        if paradigm == "recurrent" and len(x.shape) == 2:
            x.unsqueeze_(dim=1)
        batch_size, seq_len, d_model = x.shape
        # q = self.transform_into_heads(self.W_q(x))
        # k = self.transform_into_heads(self.W_k(x))
        q = self.W_q(x)
        k = self.W_k(x)
        q = self.transform_into_heads(q)
        k = self.transform_into_heads(k)
        
        print(q.shape)
        print(k.shape)
        q, k = self.rotary_embedding.rotate_queries_and_keys(q, k, offset=chunk_index)
        # q = self.xpos(q, offset=chunk_index, downscale=False)
        # k = self.xpos(k, offset=chunk_index, downscale=True)
        

        v = self.transform_into_heads(self.W_v(x))

        if paradigm == "parallel":
            decay_mask = self.get_masks(seq_len=seq_len)
            print(decay_mask)
            retention_scores = self.parallel_retention(q, k, v, decay_mask)
            current_kv = None

        elif paradigm == "recurrent":
            decay = self.gamma.view(-1, 1, 1)
            # print(self.gamma)
            # print(decay)
            retention_scores, current_kv = self.recurrent_retention(
                q,
                k,
                v,
                past_kv=past_kv,
                decay=decay,
            )
        elif paradigm == "chunkwise":
            decay_mask = self.get_masks(seq_len=seq_len)
            # inner_decay = self.get_masks(seq_len=seq_len)
            chunk_decay = self.gamma.view(self.num_heads, 1, 1) ** (chunk_index + 1)
            retention_scores, current_kv = self.chunkwise_retention(
                q,
                k,
                v,
                past_kv=past_kv,
                decay_mask=decay_mask,
                chunk_decay=chunk_decay,
                inner_decay=0,
            )
        else:
            raise NotImplementedError

        concat_scores = (
            torch.transpose(retention_scores, 1, 2)
            .flatten(-2, -1)
            .view(batch_size * seq_len, -1)
        )  # batch_size, seq_len, d_model

        concat_scores = self.group_norm_concat(concat_scores).view(
            batch_size, seq_len, -1
        )

        output = self.W_o(self.swish(self.W_g(x)) * concat_scores)
        return output, current_kv

    def parallel_retention(self, q, k, v, decay_mask):
        retention = q @ k.transpose(-1, -2)  # / torch.sqrt(torch.tensor(self.k_dim))
        retention = retention * decay_mask
        output = retention @ v
        # kv = k.transpose(-1, -2) @ v

        return output

    def recurrent_retention(self, q, k, v, past_kv, decay):
        print(past_kv)
        if past_kv is None:
            past_kv = 0
        # print(decay)

        current_kv = decay * past_kv + k.transpose(-1, -2) @ v
        output = q @ current_kv  # / torch.sqrt(torch.tensor(self.k_dim))

        return output, current_kv

    def chunkwise_retention(
        self, q, k, v, past_kv, decay_mask, chunk_decay, inner_decay
    ):
        retention = q @ k.transpose(-1, -2)
        retention = retention * decay_mask
        inner_retention = retention @ v
        cross_retention = (q @ past_kv) * inner_decay
        retention = inner_retention + cross_retention
        output = self.group_norm_retention(retention)
        current_kv = chunk_decay * past_kv + k.transpose(-1, -2) @ v
        return output, current_kv

    def transform_into_heads(self, x):
        batch_size, seq_len, d_large = x.shape
        return torch.transpose(
            x.view(batch_size, seq_len, self.num_heads, d_large // self.num_heads), 1, 2
        )  ## batch_size, num_heads, seq_len, d_kqv

    def get_masks(self, seq_len):
        row_indices = torch.arange(seq_len).view(-1, 1)
        column_indices = torch.arange(seq_len).view(1, -1)
        power_matrix = row_indices - column_indices
        gamma_expanded = self.gamma.view(-1, 1, 1)
        masks = torch.where(
            power_matrix >= 0, gamma_expanded**power_matrix, torch.zeros(1, 1)
        )
        return masks

        norm_masks = torch.div(masks, torch.sqrt(masks.sum(-1).unsqueeze(-1)))

        return norm_masks


class RetentionNetworkBlock(nn.Module):
    def __init__(
        self,
        d_model,
        k_dim,
        v_dim,
        num_heads,
        ffn_intermediate_layer_dim,
        ffn_intermediate_layers=1,
    ):
        super(RetentionNetworkBlock, self).__init__()
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

    def forward(self, x, past_kv, paradigm):
        retention_scores, current_kv = self.multi_scale_retention(
            self.layer_norm_1(x), past_kv, paradigm
        )
        y = retention_scores + x
        output = self.ffn(self.layer_norm_2(y)) + y
        return output, current_kv


class RetentionNetwork(nn.Module):
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
        super(RetentionNetwork, self).__init__()
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList(
            [
                RetentionNetworkBlock(
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

    def forward(self, x, paradigm):
        past_kvs = [None] * self.num_blocks
        if paradigm == "recurrent" and len(x.shape) == 2:
            x.unsqueeze_(dim=1)

        for i, layer in enumerate(self.layers):
            x, current_kvs = layer(x, past_kvs[i], paradigm)
        return x


if __name__ == "__main__":
    retnet = RetentionNetwork(
        d_model=256,
        k_dim=32,
        v_dim=64,
        num_heads=8,
        num_blocks=6,
        ffn_intermediate_layer_dim=1024,
        ffn_intermediate_layers=1,
    )
    random_input = torch.rand(9, 3, 256)
    parallel_output = retnet(random_input, paradigm="parallel")

    b = None
    for i in range(random_input.shape[1]):
        # print(random_input[:, i, :].shape)
        a, b = retnet(
            random_input[:, i, :].unsqueeze(1), paradigm="recurrent", past_kv=b
        )

    # rand_input = torch.rand(8, 256)
    # output = retnet(rand_input, paradigm="recurrent")
    # print(output.shape)
