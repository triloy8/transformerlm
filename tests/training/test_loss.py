import torch

from leash.objectives.loss import (
    cross_entropy,
    diffusion_cross_entropy,
    mntp_cross_entropy,
    uniform_gidd_loss,
    unweighted_diffusion_cross_entropy,
)


def test_diffusion_cross_entropy_matches_weighted_average():
    logits = torch.log(torch.tensor([
        [[0.7, 0.3], [0.2, 0.8]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, True]])
    p_mask = torch.tensor([[0.5, 1.0]])

    ce = cross_entropy(logits, targets, reduction="none")
    manual = ((ce[0, 0] / p_mask[0, 0]) + (ce[0, 1] / p_mask[0, 1])) / 2
    ours = diffusion_cross_entropy(logits, targets, mask, p_mask)
    assert torch.allclose(ours, manual)


def test_diffusion_cross_entropy_ignores_unmasked_tokens():
    logits = torch.log(torch.tensor([
        [[0.6, 0.4], [0.1, 0.9]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])
    p_mask = torch.tensor([[0.5, 0.5]])

    value = diffusion_cross_entropy(logits, targets, mask, p_mask)
    ce = cross_entropy(logits, targets, reduction="none")[0, 0]
    expected = (ce / p_mask[0, 0]) / (targets.numel())
    assert torch.allclose(value, expected)


def test_unweighted_diffusion_cross_entropy_averages_masked_tokens_only():
    logits = torch.log(torch.tensor([
        [[0.6, 0.4], [0.1, 0.9], [0.25, 0.75]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[True, False, True]])

    value = unweighted_diffusion_cross_entropy(logits, targets, mask)
    ce = cross_entropy(logits, targets, reduction="none")
    expected = (ce[0, 0] + ce[0, 2]) / 2
    assert torch.allclose(value, expected)


def test_unweighted_diffusion_cross_entropy_respects_loss_mask():
    logits = torch.log(torch.tensor([
        [[0.6, 0.4], [0.1, 0.9], [0.25, 0.75]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[True, True, True]])
    loss_mask = torch.tensor([[True, False, True]])

    value = unweighted_diffusion_cross_entropy(logits, targets, mask, loss_mask=loss_mask)
    ce = cross_entropy(logits, targets, reduction="none")
    expected = (ce[0, 0] + ce[0, 2]) / 2
    assert torch.allclose(value, expected)


def test_uniform_gidd_loss_matches_reference_formula():
    logits = torch.tensor([
        [[1.0, -0.5, 0.25], [0.1, 0.3, -0.2]],
    ])
    z_t = torch.tensor([[0, 2]])
    targets = torch.tensor([[0, 1]])
    t = torch.tensor([0.4])
    beta_is = 0.7

    value = uniform_gidd_loss(
        logits,
        z_t,
        targets,
        t,
        vocab_size=3,
        beta_is=beta_is,
        z_loss_strength=0.0,
        reduction="none",
    )

    alpha = torch.full_like(value, 0.6)
    u = torch.full_like(value, 0.4 / 3.0)
    x_hat = torch.softmax(logits, dim=-1)
    log_q_v = torch.log(alpha.unsqueeze(-1) * x_hat + u.unsqueeze(-1))
    log_q_at_x = log_q_v.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    sum_log_q = log_q_v.sum(dim=-1)
    h_ce = -alpha * log_q_at_x - u * sum_log_q
    alpha_plus_u = alpha + u
    h_q = -(alpha_plus_u * torch.log(alpha_plus_u) + 2 * u * torch.log(u))
    kl = h_ce - h_q

    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    q_zt_x = alpha * (z_t == targets).to(alpha.dtype) + u
    q_zt_x_hat = alpha * x_hat_at_zt + u
    log_ratio = torch.log(q_zt_x) - torch.log(q_zt_x_hat)
    is_div = torch.exp(log_ratio) - log_ratio - 1.0
    w_kept = (1.0 - alpha) / (1.0 + 2 * alpha)
    w = torch.where(z_t == targets, w_kept, torch.ones_like(w_kept))
    expected = w * kl + beta_is * w * is_div

    assert torch.allclose(value, expected)


def test_uniform_gidd_loss_uses_loss_and_reduction_masks():
    logits = torch.tensor([
        [[0.4, -0.1], [0.3, 0.2], [-0.5, 0.6]],
    ])
    z_t = torch.tensor([[0, 1, 0]])
    targets = torch.tensor([[0, 0, 1]])
    t = torch.tensor([0.5])
    loss_mask = torch.tensor([[True, True, False]])
    reduction_mask = torch.tensor([[True, False, True]])

    per_token = uniform_gidd_loss(
        logits,
        z_t,
        targets,
        t,
        vocab_size=2,
        z_loss_strength=0.0,
        reduction="none",
    )
    value = uniform_gidd_loss(
        logits,
        z_t,
        targets,
        t,
        vocab_size=2,
        z_loss_strength=0.0,
        loss_mask=loss_mask,
        reduction_mask=reduction_mask,
    )

    assert torch.allclose(value, per_token[:, :1].mean())


def test_mntp_cross_entropy_uses_shifted_targets_and_masks():
    logits = torch.log(torch.tensor([
        [[0.6, 0.4], [0.2, 0.8], [0.9, 0.1]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[True, True, False]])
    p_mask = torch.tensor([[0.5]])

    value = mntp_cross_entropy(logits, targets, mask, p_mask)

    shifted_logits = logits[:, :-1, :]
    shifted_targets = targets[:, 1:]
    shifted_mask = mask[:, :-1]
    expected = diffusion_cross_entropy(shifted_logits, shifted_targets, shifted_mask, p_mask)
    assert torch.allclose(value, expected)
