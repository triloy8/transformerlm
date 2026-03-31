import torch

from leash.inference.sampling import top_p_filter, softmax, compute_transfer_schedule


def test_softmax_matches_torch(device):
    logits = torch.tensor([[3.0, 1.0, -2.0]], device=device)
    ours = softmax(logits, dim=-1)
    torch_softmax = torch.nn.functional.softmax(logits, dim=-1)
    assert torch.allclose(ours, torch_softmax, atol=1e-6)


def test_top_p_filter_normalization(device):
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device)
    filtered = top_p_filter(probs, p=0.7)
    assert torch.allclose(filtered.sum(dim=-1), torch.ones(1, device=device))
    assert (filtered > 0).sum().item() == 2


def test_top_p_filter_extremes(device):
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device)
    argmax = top_p_filter(probs, p=0.0)
    assert argmax.argmax(dim=-1).item() == 0
    assert torch.allclose(argmax.sum(dim=-1), torch.ones(1, device=device))

    unchanged = top_p_filter(probs, p=1.0)
    assert torch.allclose(unchanged, probs, atol=1e-7)


def test_compute_transfer_schedule_sum_equals_mask_count(device):
    mask = torch.tensor([[True, False, True, False, True]], dtype=torch.bool)
    schedule = compute_transfer_schedule(mask, steps=2)
    assert schedule.shape == (1, 2)
    assert int(schedule.sum()) == int(mask.sum())
