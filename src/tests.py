import unittest
import torch
import random
from modules import GatedMultiScaleRetentionLayer


class TestGatedMultiScaleRetentionLayer(unittest.TestCase):
    # def test_input_shapes_parallel_retention(self):
    #     d_model = 512
    #     k_dim = 256
    #     v_dim = 512
    #     num_heads = 16
    #     msr = GatedMultiScaleRetentionLayer(
    #         d_model=d_model, k_dim=k_dim, v_dim=v_dim, num_heads=num_heads
    #     )

    #     batch_size = 3
    #     seq_len = random.randint(0, 32)

    #     x = torch.rand(batch_size, seq_len, d_model)
    #     output, current_kv = msr(x, paradigm="parallel")

    #     self.assertEqual(x.shape, output.shape)
    #     self.assertIsNone(current_kv)

    # def test_input_shapes_recurrent_retention(self):
    #     d_model = 256
    #     k_dim = 32
    #     v_dim = 64
    #     num_heads = 16
    #     msr = GatedMultiScaleRetentionLayer(
    #         d_model=d_model, k_dim=k_dim, v_dim=v_dim, num_heads=num_heads
    #     )

    #     batch_size = 3
    #     seq_len = random.randint(1, 32)

    #     x = torch.rand(batch_size, seq_len, d_model)

    #     output_list = []
    #     past_kv = None

    #     for i in range(seq_len):
    #         output, past_kv = msr(x[:, i, :], paradigm="recurrent", past_kv=past_kv)
    #         output_list.append(output)

    #     output = torch.cat(output_list, dim=1)

    #     self.assertEqual(output.shape, x.shape)

    # def test_parallel_retention(self):
    #     d_model = 512
    #     k_dim = 256
    #     v_dim = 512
    #     num_heads = 16
    #     msr = GatedMultiScaleRetentionLayer(
    #         d_model=d_model, k_dim=k_dim, v_dim=v_dim, num_heads=num_heads
    #     )

    #     batch_size = 3
    #     seq_len = 19
    #     sub_len = 11

    #     x = torch.rand(batch_size, seq_len, d_model)
    #     output, current_kv = msr(x, paradigm="parallel")

    #     y, _ = msr(x[:, :sub_len, :], paradigm="parallel")

    #     torch.testing.assert_close(output[:, :sub_len, :], y)

    def test_paradigms_equivalence(self):
        d_model = 32
        k_dim = 32
        v_dim = 64
        num_heads = 1

        msr = GatedMultiScaleRetentionLayer(
            d_model=d_model, k_dim=k_dim, v_dim=v_dim, num_heads=num_heads
        )

        batch_size = 1
        seq_len = 2

        x = torch.rand(batch_size, seq_len, d_model)

        parallel_output, _ = msr(x, paradigm="parallel")

        past_kv = None
        output_list = []

        for i in range(seq_len):
            output, past_kv = msr(x[:, i, :], paradigm="recurrent", past_kv=past_kv, chunk_index=i)
            # print(past_kv.shape)

            # print(output)
            # print("-----------------------------------------")
            output_list.append(output)

        recursive_output = torch.cat(output_list, dim=1)
        # print(parallel_output)

        # print(recursive_output)

        torch.testing.assert_close(parallel_output, recursive_output)
        # self.assertEqual(output.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
