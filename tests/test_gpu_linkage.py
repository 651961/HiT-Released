import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA linkage kernel")
def test_symmetric_row_loads_match_column_loads_bitwise():
    from hnet.runtime.gpu_linkage import raw_linkage_gpu

    generator = torch.Generator(device="cuda").manual_seed(761)
    scores = torch.randn(4, 129, 129, device="cuda", generator=generator)
    affinity = (scores + scores.mT) * 0.125
    affinity[1].fill_(1)
    affinity[2] = affinity[2].mul(8).round().div(8)
    lengths = torch.tensor([1, 37, 128, 129], device="cuda")
    reference = raw_linkage_gpu(affinity, lengths)
    actual = raw_linkage_gpu(affinity, lengths, symmetric=True)
    assert torch.equal(actual.view(torch.int64), reference.view(torch.int64))
