"""Factored mechanics world-model components (Phase 2).

Pixel-only Lagrangian + Rayleigh dissipation branch that composes with an
AR(1) nuisance latent to produce the full transition used by
``MechanicsWorldModel``. See ``docs/models/mechanics_world_model.md`` for the
design rationale and how the pieces plug into the existing imagination-
rollout / reward-head / CEM infrastructure from Phase 0/1.
"""

from models.mechanics.encoder import (
    FactoredEncoder,
    FactoredPosterior,
    derive_qdot_from_q,
)
from models.mechanics.integrator import semi_implicit_euler_step
from models.mechanics.losses import (
    compute_factored_kl,
    compute_mechanics_world_model_loss,
)
from models.mechanics.lagrangian import (
    Actuation,
    Dissipation,
    LagrangianDynamics,
    MassMatrix,
    Potential,
    forward_acceleration,
    kinetic_energy,
    total_energy,
)
from models.mechanics.transition import MechanicsPrior, MechanicsTransition
from models.mechanics.world_model import (
    MechanicsRollout,
    MechanicsWorldModel,
    MechanicsWorldModelOutput,
)

__all__ = [
    "Actuation",
    "compute_factored_kl",
    "compute_mechanics_world_model_loss",
    "derive_qdot_from_q",
    "Dissipation",
    "FactoredEncoder",
    "FactoredPosterior",
    "LagrangianDynamics",
    "MassMatrix",
    "MechanicsPrior",
    "MechanicsRollout",
    "MechanicsTransition",
    "MechanicsWorldModel",
    "MechanicsWorldModelOutput",
    "Potential",
    "forward_acceleration",
    "kinetic_energy",
    "semi_implicit_euler_step",
    "total_energy",
]
