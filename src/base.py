import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import ExtrapolatablePositionEmbedding

"""
Summary:
1) X: (batch_size, seq_len, d)
2) W_Q:

Mask D-> upper triangular matrix

"""
# TODO: Normalize Masks


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_layer_dim: int,
        output_dim: int,
        num_intermediate_layers: int = 1,
        activation: str = "gelu",
    ) -> None:
        """
        :param self: Represent the instance of the class
        :param input_dim: int: Specify the size of the input to the first layer
        :param intermediate_layer_dim: int: Specify the dimension of the intermediate layers
        :param output_dim: int: Specify the output dimension of the feed forward network
        :param num_intermediate_layers: int: Specify the number of intermediate layers in the network
        :param activation: str: Specify which activation function to use
        :return: None
        """
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, intermediate_layer_dim)]
            + [nn.Linear(intermediate_layer_dim, intermediate_layer_dim)]
            * (num_intermediate_layers - 1)
        )
        self.output_layer = nn.Linear(intermediate_layer_dim, output_dim)
        if activation == "gelu":
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward function of the model.

        :param self: Access variables that belong to the class
        :param x: torch.Tensor: Pass in the input data
        :return: The output of the last layer
        """

        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class GatedMultiScaleRetentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        k_dim: int,
        v_dim: int,
        num_heads: int,
        rotation_gamma: float = 0.4,
    ) -> None:
        """
        :param self: Represent the instance of the class
        :param d_model: int: Set the dimension of the model
        :param k_dim: int: Specify the dimension of the key vector
        :param v_dim: int: Set the dimension of the value vector
        :param num_heads: int: Determine the number of heads in the multi-head attention
        :return: None
        """

        super(GatedMultiScaleRetentionLayer, self).__init__()
        self.gamma = 1 - torch.exp(
            (-5 - torch.arange(0, num_heads)) * torch.log(torch.tensor(2))
        )  # shape: [num_heads]

        self.W_q = nn.Linear(d_model, k_dim * num_heads)
        self.W_k = nn.Linear(d_model, k_dim * num_heads)
        self.W_v = nn.Linear(d_model, v_dim * num_heads)
        self.num_heads = num_heads
        self.xpos = ExtrapolatablePositionEmbedding(
            d=num_heads * k_dim, gamma=rotation_gamma, debug=False
        )
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
        q = self.W_q(x)
        k = self.W_k(x)

        q, k = self.xpos.rotate_key_and_query(q, k, offset=chunk_index)

        q = self.transform_into_heads(q)
        k = self.transform_into_heads(k)
        v = self.transform_into_heads(self.W_v(x))

        if paradigm == "parallel":
            decay_mask = self.get_masks(seq_len=seq_len)
            retention_scores = self.parallel_retention(q, k, v, decay_mask)
            current_kv = None

        elif paradigm == "recurrent":
            decay = self.gamma.view(-1, 1, 1)
            retention_scores, current_kv = self.recurrent_retention(
                q,
                k,
                v,
                past_kv=past_kv,
                decay=decay,
            )

        elif paradigm == "chunkwise":
            decay_mask = self.get_masks(seq_len=seq_len)
            inner_decay = self.gamma.unsqueeze(1) ** (torch.arange(seq_len) + 1)
            chunk_decay = self.gamma.view(-1, 1, 1) ** (seq_len)
            retention_scores, current_kv = self.chunkwise_retention(
                q,
                k,
                v,
                past_kv=past_kv,
                decay_mask=decay_mask,
                chunk_decay=chunk_decay,
                inner_decay=inner_decay,
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
        retention = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(self.k_dim))
        retention = retention * decay_mask
        retention = retention.div(
            torch.max(torch.abs(retention.sum(-1)), torch.tensor(1)).unsqueeze(-1)
        )
        output = retention @ v
        # kv = k.transpose(-1, -2) @ v

        return output

    def recurrent_retention(self, q, k, v, past_kv, decay):
        if past_kv is None:
            past_kv = 0

        current_kv = decay * past_kv + k.transpose(-1, -2) @ v
        output = q @ current_kv / torch.sqrt(torch.tensor(self.k_dim))

        return output, current_kv

    def chunkwise_retention(
        self, q, k, v, past_kv, decay_mask, chunk_decay, inner_decay
    ):
        _, _, seq_len, _ = q.shape
        retention = q @ k.transpose(-1, -2) / torch.sqrt(torch.tensor(self.k_dim))
        retention = retention * decay_mask

        retention = retention.div(
            torch.max(torch.abs(retention.sum(-1)), torch.tensor(1)).unsqueeze(-1)
        )

        inner_retention = retention @ v

        if past_kv is None:
            past_kv = 0
            cross_retention = 0
        else:
            cross_retention = (
                (q @ past_kv)
                * inner_decay.view(1, self.num_heads, seq_len, 1)
                / torch.sqrt(torch.tensor(self.k_dim))
            )

        retention = inner_retention + cross_retention
        current_kv = chunk_decay * past_kv + (
            k.unsqueeze(-1)
            * torch.flip(inner_decay / (self.gamma.unsqueeze(1)), dims=(1,)).view(
                1, self.num_heads, seq_len, 1, 1
            )
            * v.unsqueeze(-2)
        ).sum(2)

        return retention, current_kv

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
        ## Normalize Masks
        # norm_masks = torch.div(masks, torch.sqrt(masks.sum(-1).unsqueeze(-1)))
        # return norm_masks
