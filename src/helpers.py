import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtrapolatablePositionEmbedding(nn.Module):
    """
    This is an implementation of relative positional embedding xPos introduced in paper,
    A Length-Extrapolatable Transformer,2022.
    """

    def __init__(self, d: int, gamma: torch.Tensor, debug: bool = False) -> None:
        """
        :param self: Represent the instance of the class
        :param d: int: Specify the dimension of the num_heads * k_dim
        :param gamma: torch.Tensor: Scale the position key, query
        :param debug: bool: Debug flag
        :return: None
        """
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

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        The rotate function takes a tensor x and rotates it.
        [x0, x1, x2, x2, ...] -> [−x1, x0, −x3, x2, ...]
        :param self: Allow an object to refer to itself inside of a method
        :param x: Pass in the input tensor
        :return: A rotated version of the input tensor
        """

        result = x.clone()
        result[:, :, 0::2] = -x[:, :, 1::2]
        result[:, :, 1::2] = x[:, :, 0::2]
        return result

    def rotate_key_and_query(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """
        The rotate_key_and_query function is used to rotate the key and query vectors.
        The rotation is done by multiplying the key and query vectors with a cosine matrix,
        and then adding them to their rotated versions multiplied by a sine matrix. The
        rotated version of each vector is obtained by shifting it one position to the right.

        :param self: Represent the object itself
        :param q: Represent the query matrix
        :param k: Represent the key matrix
        :param offset: Shift the position of the sin and cosine functions
        :return: The query and key vectors after applying the rotation operation
        """
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
