import pytest
import torch

from bioblp.benchmarking.featurise import apply_common_mask


class TestApplyCommonMask:

    data_A = torch.arange(0., 9.).resize(3, 3)
    data_B = torch.arange(9., 21.).resize(3, 4)

    labels = torch.ones(3)

    def test_mask_consistency(self):
        mask_A = torch.tensor([0, 1])
        mask_B = torch.tensor([0, 1, 2])

        inputs = [("A", self.data_A, mask_A), ("B", self.data_B, mask_B)]

        masked_inputs, _ = apply_common_mask(inputs, labels=self.labels)

        assert masked_inputs[0][1].size(0) == len(mask_A)
        assert masked_inputs[0][1].size(0) == masked_inputs[1][1].size(0)

    def test_mask_consistency_labels(self):
        mask_A = torch.tensor([0, 2])
        mask_B = torch.tensor([0, 1, 2])

        labels = torch.tensor([1, 1, 0])
        expected_labels = torch.tensor([1, 0])

        inputs = [("A", self.data_A, mask_A), ("B", self.data_B, mask_B)]

        _, masked_labels = apply_common_mask(inputs, labels=labels)

        assert len(masked_labels) == len(mask_A)
        assert torch.sum((masked_labels - expected_labels)) == 0
