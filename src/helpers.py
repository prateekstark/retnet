import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtrapolatablePositionEmbedding(nn.Module):
    def __init__(self, d: int, gamma: torch.Tensor, debug: bool = False) -> None:
        super(ExtrapolatablePositionEmbedding, self).__init__()
        if debug:
            assert d % 2 == 0

        self.theta = torch.exp(
            ((-4 / d) * torch.arange(0, d, 2)) * torch.log(torch.tensor(10))
        )  ##d//2

        if debug:
            assert self.theta.shape[0] == d // 2
        self.gamma = gamma
        self.scale = ((torch.arange(0, d, 2) / d) + self.gamma) / (
            1 + self.gamma
        )  ##d // 2

        if debug:
            assert self.scale.shape[0] == d // 2

    def rotate(self, x):
        result = x.clone()
        result[:, :, 0::2] = -x[:, :, 1::2]
        result[:, :, 1::2] = x[:, :, 0::2]
        return result

    def rotate_key_and_query(self, q, k, offset=0):
        _, seq_len, _ = q.shape

        cos = torch.cos(
            (torch.arange(0, seq_len) + offset).unsqueeze(dim=1) * self.theta
        ).repeat_interleave(2, dim=-1)

        sin = torch.sin(
            (torch.arange(0, seq_len) + offset).unsqueeze(dim=1) * self.theta
        ).repeat_interleave(2, dim=-1)

        T = (
            self.scale ** ((torch.arange(0, seq_len) + offset).unsqueeze(1))
        ).repeat_interleave(2, dim=-1)

        return ((q * cos + self.rotate(q) * sin)) * T, (
            (k * cos + self.rotate(k) * sin)
        ) / T
