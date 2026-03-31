from leash.objectives.base import Objective
from leash.objectives.data import DiffusionBatch, AutoregressiveBatch, CategoricalFlowBatch, get_batch, get_autoregressive_batch
from leash.objectives.loss import cross_entropy, diffusion_cross_entropy, mntp_cross_entropy, autoregressive_cross_entropy
from leash.objectives.diffusion import DiffusionObjective, FlowMatchingObjective, MegaDlmDiffusionObjective
from leash.objectives.autoregressive import AutoregressiveObjective
from leash.objectives.joint import JointDiffusionAutoregressiveObjective, JointMntpAutoregressiveObjective
from leash.objectives.categorical_flow import CategoricalFlowObjective


def build_objective(cfg, tokenizer) -> Objective:
    name = str(getattr(cfg, "training_objective", "diffusion")).lower()
    if name == "ar":
        return AutoregressiveObjective(cfg, tokenizer)
    if name == "megadlm-diffusion":
        return MegaDlmDiffusionObjective(cfg, tokenizer)
    if name == "flow":
        return FlowMatchingObjective(cfg, tokenizer)
    if name == "categorical-flow":
        return CategoricalFlowObjective(cfg, tokenizer)
    if name == "joint-diffusion-ar":
        return JointDiffusionAutoregressiveObjective(cfg, tokenizer)
    if name == "joint-mntp-ar":
        return JointMntpAutoregressiveObjective(cfg, tokenizer)
    return DiffusionObjective(cfg, tokenizer)


__all__ = [
    "Objective",
    "DiffusionObjective",
    "MegaDlmDiffusionObjective",
    "FlowMatchingObjective",
    "AutoregressiveObjective",
    "JointDiffusionAutoregressiveObjective",
    "JointMntpAutoregressiveObjective",
    "CategoricalFlowObjective",
    "build_objective",
    "DiffusionBatch",
    "AutoregressiveBatch",
    "CategoricalFlowBatch",
    "get_batch",
    "get_autoregressive_batch",
    "cross_entropy",
    "diffusion_cross_entropy",
    "mntp_cross_entropy",
    "autoregressive_cross_entropy",
]
