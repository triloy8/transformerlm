from leash.objectives.base import Objective
from leash.objectives.data import DiffusionBatch, AutoregressiveBatch, CategoricalFlowBatch, get_batch, get_autoregressive_batch
from leash.objectives.loss import (
    autoregressive_cross_entropy,
    cross_entropy,
    diffusion_cross_entropy,
    mntp_cross_entropy,
    uniform_gidd_loss,
    unweighted_diffusion_cross_entropy,
)
from leash.objectives.diffusion import (
    DiffusionObjective,
    FlowMatchingObjective,
    MegaDlmDiffusionObjective,
    SumiUniformGiddDiffusionObjective,
    UniformStateDiffusionObjective,
)
from leash.objectives.autoregressive import AutoregressiveObjective
from leash.objectives.joint import JointDiffusionAutoregressiveObjective, JointMntpAutoregressiveObjective
from leash.objectives.categorical_flow import CategoricalFlowObjective
from leash.objectives.registry import build_objective, get_objective_factory, list_objectives, register_objective


__all__ = [
    "Objective",
    "DiffusionObjective",
    "MegaDlmDiffusionObjective",
    "SumiUniformGiddDiffusionObjective",
    "UniformStateDiffusionObjective",
    "FlowMatchingObjective",
    "AutoregressiveObjective",
    "JointDiffusionAutoregressiveObjective",
    "JointMntpAutoregressiveObjective",
    "CategoricalFlowObjective",
    "build_objective",
    "get_objective_factory",
    "list_objectives",
    "register_objective",
    "DiffusionBatch",
    "AutoregressiveBatch",
    "CategoricalFlowBatch",
    "get_batch",
    "get_autoregressive_batch",
    "cross_entropy",
    "diffusion_cross_entropy",
    "unweighted_diffusion_cross_entropy",
    "uniform_gidd_loss",
    "mntp_cross_entropy",
    "autoregressive_cross_entropy",
]
