from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence


#============CONSTANTS===============
MIN_NORMALIZED_VALUE = 0.0
MAX_NORMALIZED_VALUE = 1.0
DEFAULT_TIMESTEP = 0.05
DEFAULT_BASE_WEIGHT = 0.0
DEFAULT_RECOVERY_RATE = 0.1
DEFAULT_ADAPTATION_RATE = 0.01
DEFAULT_SIGMOID_STEEPNESS = 8.0
DEFAULT_SIGMOID_MIDPOINT = 0.5
DEFAULT_HILL_COEFFICIENT = 2.0
DEFAULT_HILL_HALF_SATURATION = 0.5
DEFAULT_EXPONENTIAL_RATE = 3.0
DEFAULT_METASTABILITY_GAIN = 4.0
DEFAULT_METASTABILITY_MIDPOINT = 0.5


#============HELPER FUNCTIONS===============
def _validate_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float, not bool.")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite; received {numeric_value!r}.")
    return numeric_value


def _clamp(value: float, minimum: float = MIN_NORMALIZED_VALUE, maximum: float = MAX_NORMALIZED_VALUE) -> float:
    return max(minimum, min(maximum, value))


def _sigmoid(value: float, steepness: float = DEFAULT_SIGMOID_STEEPNESS, midpoint: float = DEFAULT_SIGMOID_MIDPOINT) -> float:
    exponent = -steepness * (value - midpoint)
    return 1.0 / (1.0 + math.exp(exponent))


#============ABSTRACT CLASSES AND PROTOCOLS===============

class HomeostasisInput(Protocol):
    """Protocol for subsystem inputs that perturb graph dynamics."""

    def perturbations(self, state: HomeostasisState) -> Mapping[str, float]:
        """Return finite additive perturbations by variable name."""
        ...


class TransferFunction(ABC):
    """Base class for nonlinear edge transfer functions."""

    @abstractmethod
    def __call__(self, value: float) -> float:
        """Transform a source-node value into an edge signal."""


class AdaptiveRule(ABC):
    """Base class for deterministic edge-weight adaptation rules."""

    @abstractmethod
    def update(self, weight: float, source_value: float, destination_value: float, dt: float) -> float:
        """Return the next deterministic edge weight."""


class Integrator(ABC):
    """Base class for numerical homeostasis integrators."""

    @abstractmethod
    def integrate(
        self,
        state: HomeostasisState,
        derivative: GraphDerivative,
        dt: float,
    ) -> HomeostasisState:
        """Integrate the derivative over dt and return the next state."""


#============CONCRETE IMPLEMENTATIONS & DATA STRUCTURES===============

class HomeostasisVariable(str, Enum):
    ENERGY = "energy"
    STRESS = "stress"
    CURIOSITY = "curiosity"
    COGNITIVE_LOAD = "cognitive_load"
    SOCIAL_NEED = "social_need"
    METASTABILITY = "metastability"


@dataclass(frozen=True, slots=True)
class HomeostasisState:

    values: Mapping[str, float]
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        validated = {
            name: _clamp(_validate_finite(value, f"values[{name!r}]"))
            for name, value in self.values.items()
        }
        if not validated:
            raise ValueError("HomeostasisState requires at least one variable.")
        if not isinstance(self.last_updated, datetime):
            raise TypeError("last_updated must be a datetime instance.")
        object.__setattr__(self, "values", MappingProxyType(validated))

    def value(self, variable: str | HomeostasisVariable) -> float:
        key = variable.value if isinstance(variable, HomeostasisVariable) else variable
        try:
            return self.values[key]
        except KeyError as exc:
            raise KeyError(f"Unknown homeostasis variable: {key}") from exc


@dataclass(frozen=True, slots=True)
class EmotionInput:
    perturbation: Mapping[str, float] = field(default_factory=dict)
    def perturbations(self, state: HomeostasisState) -> Mapping[str, float]:
        return self.perturbation


@dataclass(frozen=True, slots=True)
class ReasoningInput:
    perturbation: Mapping[str, float] = field(default_factory=dict)
    def perturbations(self, state: HomeostasisState) -> Mapping[str, float]:
        return self.perturbation


@dataclass(frozen=True, slots=True)
class MemoryInput:
    perturbation: Mapping[str, float] = field(default_factory=dict)
    def perturbations(self, state: HomeostasisState) -> Mapping[str, float]:
        return self.perturbation


@dataclass(frozen=True, slots=True)
class EnvironmentInput:
    perturbation: Mapping[str, float] = field(default_factory=dict)
    def perturbations(self, state: HomeostasisState) -> Mapping[str, float]:
        return self.perturbation


@dataclass(frozen=True, slots=True)
class SleepInput:
    perturbation: Mapping[str, float] = field(default_factory=dict)
    def perturbations(self, state: HomeostasisState) -> Mapping[str, float]:
        return self.perturbation


@dataclass(frozen=True, slots=True)
class GraphNode:
    name: str
    equilibrium: float
    recovery_rate: float = DEFAULT_RECOVERY_RATE

    def __post_init__(self) -> None:
        object.__setattr__(self, "equilibrium", _clamp(_validate_finite(self.equilibrium, "equilibrium")))
        recovery_rate = _validate_finite(self.recovery_rate, "recovery_rate")
        if recovery_rate < 0.0:
            raise ValueError("recovery_rate must be non-negative.")
        object.__setattr__(self, "recovery_rate", recovery_rate)
        if not self.name:
            raise ValueError("GraphNode.name cannot be empty.")


@dataclass(frozen=True, slots=True)
class TanhTransfer(TransferFunction):
    gain: float = 1.0
    midpoint: float = DEFAULT_SIGMOID_MIDPOINT

    def __call__(self, value: float) -> float:
        """Return tanh(gain * (value - midpoint))."""
        return math.tanh(self.gain * (_validate_finite(value, "value") - self.midpoint))


@dataclass(frozen=True, slots=True)
class SigmoidTransfer(TransferFunction):
    steepness: float = DEFAULT_SIGMOID_STEEPNESS
    midpoint: float = DEFAULT_SIGMOID_MIDPOINT

    def __call__(self, value: float) -> float:
        return _sigmoid(_validate_finite(value, "value"), self.steepness, self.midpoint)


@dataclass(frozen=True, slots=True)
class HillTransfer(TransferFunction):
    coefficient: float = DEFAULT_HILL_COEFFICIENT
    half_saturation: float = DEFAULT_HILL_HALF_SATURATION

    def __call__(self, value: float) -> float:
        x = _clamp(_validate_finite(value, "value"))
        numerator = x**self.coefficient
        denominator = self.half_saturation**self.coefficient + numerator
        if denominator == 0.0:
            return 0.0
        return numerator / denominator


@dataclass(frozen=True, slots=True)
class ExponentialTransfer(TransferFunction):
    rate: float = DEFAULT_EXPONENTIAL_RATE
    def __call__(self, value: float) -> float:
        return 1.0 - math.exp(-self.rate * _clamp(_validate_finite(value, "value")))


class LinearTransfer(TransferFunction):
    """Linear transfer intended for deterministic testing and calibration."""

    def __call__(self, value: float) -> float:
        """Return the finite input value unchanged."""
        return _validate_finite(value, "value")


class StaticAdaptiveRule(AdaptiveRule):
    """Adaptive rule that keeps an edge weight unchanged."""

    def update(self, weight: float, source_value: float, destination_value: float, dt: float) -> float:
        """Return the current edge weight unchanged."""
        _validate_finite(source_value, "source_value")
        _validate_finite(destination_value, "destination_value")
        _validate_finite(dt, "dt")
        return _validate_finite(weight, "weight")


@dataclass(frozen=True, slots=True)
class CorrelationAdaptiveRule(AdaptiveRule):
    """Deterministic Hebbian-style rule with bounded weight updates."""

    rate: float = DEFAULT_ADAPTATION_RATE
    minimum_weight: float = -1.0
    maximum_weight: float = 1.0

    def update(self, weight: float, source_value: float, destination_value: float, dt: float) -> float:
        """Move weight according to centered source-destination correlation."""
        current_weight = _validate_finite(weight, "weight")
        source = _validate_finite(source_value, "source_value") - DEFAULT_SIGMOID_MIDPOINT
        destination = _validate_finite(destination_value, "destination_value") - DEFAULT_SIGMOID_MIDPOINT
        timestep = _validate_finite(dt, "dt")
        next_weight = current_weight + self.rate * source * destination * timestep
        return _clamp(next_weight, self.minimum_weight, self.maximum_weight)


@dataclass(slots=True)
class GraphEdge:
    """Directed weighted graph edge with local transfer and adaptation behavior."""

    source: str
    destination: str
    weight: float = DEFAULT_BASE_WEIGHT
    transfer: TransferFunction = field(default_factory=LinearTransfer)
    adaptive_rule: AdaptiveRule = field(default_factory=StaticAdaptiveRule)

    def __post_init__(self) -> None:
        self.weight = _validate_finite(self.weight, "weight")
        if not self.source or not self.destination:
            raise ValueError("GraphEdge source and destination cannot be empty.")

    def signal(self, source_value: float) -> float:
        """Return this edge's weighted nonlinear signal."""
        return self.weight * self.transfer(source_value)

    def adapt(self, state: HomeostasisState, dt: float) -> None:
        """Deterministically update this edge's adaptive weight."""
        self.weight = self.adaptive_rule.update(
            self.weight,
            state.value(self.source),
            state.value(self.destination),
            dt,
        )


@dataclass(frozen=True, slots=True)
class GraphDerivative:
    """Discrete derivative components for one graph dynamics evaluation."""

    values: Mapping[str, float]
    interaction: Mapping[str, float]
    restoration: Mapping[str, float]
    external: Mapping[str, float]


class GraphDynamics:
    """Computes nonlinear graph dynamics for a homeostasis state."""

    def __init__(
        self,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        metastability_gain: float = DEFAULT_METASTABILITY_GAIN,
        metastability_midpoint: float = DEFAULT_METASTABILITY_MIDPOINT,
    ) -> None:
        if not nodes:
            raise ValueError("GraphDynamics requires at least one node.")
        self._nodes = {node.name: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise ValueError("GraphDynamics node names must be unique.")
        self._edges = list(edges)
        self._metastability_gain = _validate_finite(metastability_gain, "metastability_gain")
        self._metastability_midpoint = _validate_finite(metastability_midpoint, "metastability_midpoint")
        self._validate_edges()

    @property
    def node_names(self) -> tuple[str, ...]:
        """Return configured graph node names."""
        return tuple(self._nodes.keys())

    def derivative(
        self,
        state: HomeostasisState,
        inputs: Sequence[HomeostasisInput] = (),
    ) -> GraphDerivative:
        """
        Evaluate dH/dt = -R(H-Heq) + GraphInteraction(H) + ExternalInputs(U).

        Args:
            state: Current immutable homeostasis state.
            inputs: Structured subsystem inputs that perturb dynamics.

        Returns:
            Derivative and component maps for each configured variable.
        """
        self._validate_state_shape(state)
        interaction = {name: 0.0 for name in self._nodes}
        for edge in self._edges:
            interaction[edge.destination] += edge.signal(state.value(edge.source))

        restoration = {
            name: -node.recovery_rate * (state.value(name) - node.equilibrium)
            for name, node in self._nodes.items()
        }
        external = self._external_perturbations(state, inputs)
        derivative = {
            name: restoration[name] + interaction[name] + external[name]
            for name in self._nodes
        }
        derivative[HomeostasisVariable.METASTABILITY.value] = self._metastability(derivative)
        return GraphDerivative(
            values=derivative,
            interaction=interaction,
            restoration=restoration,
            external=external,
        )

    def adapt_edges(self, state: HomeostasisState, dt: float) -> None:
        """Apply deterministic adaptive rules to every graph edge."""
        timestep = _validate_finite(dt, "dt")
        if timestep <= 0.0:
            raise ValueError("dt must be positive.")
        for edge in self._edges:
            edge.adapt(state, timestep)

    def _metastability(self, derivative: Mapping[str, float]) -> float:
        non_meta_activity = sum(
            abs(value)
            for name, value in derivative.items()
            if name != HomeostasisVariable.METASTABILITY.value
        )
        return _sigmoid(non_meta_activity, self._metastability_gain, self._metastability_midpoint)

    def _external_perturbations(
        self,
        state: HomeostasisState,
        inputs: Sequence[HomeostasisInput],
    ) -> dict[str, float]:
        perturbations = {name: 0.0 for name in self._nodes}
        for subsystem_input in inputs:
            for name, value in subsystem_input.perturbations(state).items():
                if name not in perturbations:
                    raise KeyError(f"Input perturbation references unknown variable: {name}")
                if name == HomeostasisVariable.METASTABILITY.value:
                    raise ValueError("Metastability is computed from graph dynamics and cannot be externally perturbed.")
                perturbations[name] += _validate_finite(value, f"perturbation[{name!r}]")
        return perturbations

    def _validate_edges(self) -> None:
        for edge in self._edges:
            if edge.source not in self._nodes:
                raise KeyError(f"Edge source is not a configured node: {edge.source}")
            if edge.destination not in self._nodes:
                raise KeyError(f"Edge destination is not a configured node: {edge.destination}")

    def _validate_state_shape(self, state: HomeostasisState) -> None:
        missing = set(self._nodes) - set(state.values)
        if missing:
            raise ValueError(f"State is missing configured variables: {sorted(missing)}")


class EulerIntegrator(Integrator):
    """Euler integration for version-one homeostasis dynamics."""

    def integrate(
        self,
        state: HomeostasisState,
        derivative: GraphDerivative,
        dt: float,
    ) -> HomeostasisState:
        """Return state(t + dt) using clamped Euler integration."""
        timestep = _validate_finite(dt, "dt")
        if timestep <= 0.0:
            raise ValueError("dt must be positive.")
        values = {
            name: _clamp(state.value(name) + timestep * derivative.values[name])
            for name in state.values
        }
        values[HomeostasisVariable.METASTABILITY.value] = _clamp(
            derivative.values[HomeostasisVariable.METASTABILITY.value]
        )
        return HomeostasisState(values=values, last_updated=datetime.utcnow())


class HomeostasisEngine:
    def __init__(
        self,
        dynamics: GraphDynamics | None = None,
        integrator: Integrator | None = None,
        initial_state: HomeostasisState | None = None,
        dt: float = DEFAULT_TIMESTEP,
    ) -> None:
        self._dt = _validate_finite(dt, "dt")
        if self._dt <= 0.0:
            raise ValueError("dt must be positive.")
        self._dynamics = dynamics if dynamics is not None else self._default_dynamics()
        self._integrator = integrator if integrator is not None else EulerIntegrator()
        self._state = initial_state if initial_state is not None else self._equilibrium_state(self._dynamics)
        self._validate_state_matches_graph(self._state)

    def get_state(self) -> HomeostasisState:
        return self._state

    def evolve(
        self,
        inputs: Sequence[HomeostasisInput] = (),
        dt: float | None = None,
    ) -> HomeostasisState:
        timestep = self._dt if dt is None else _validate_finite(dt, "dt")
        if timestep <= 0.0:
            raise ValueError("dt must be positive.")
        derivative = self._dynamics.derivative(self._state, inputs)
        next_state = self._integrator.integrate(self._state, derivative, timestep)
        self._dynamics.adapt_edges(next_state, timestep)
        self._state = next_state
        return self._state

    def reset(self, state: HomeostasisState | None = None) -> HomeostasisState:
        next_state = state if state is not None else self._equilibrium_state(self._dynamics)
        self._validate_state_matches_graph(next_state)
        self._state = next_state
        return self._state

    def _validate_state_matches_graph(self, state: HomeostasisState) -> None:
        missing = set(self._dynamics.node_names) - set(state.values)
        if missing:
            raise ValueError(f"Initial state is missing graph variables: {sorted(missing)}")

    def _equilibrium_state(self, dynamics: GraphDynamics) -> HomeostasisState:
        nodes = dynamics._nodes
        return HomeostasisState(
            values={name: node.equilibrium for name, node in nodes.items()},
            last_updated=datetime.utcnow(),
        )

    def _default_dynamics(self) -> GraphDynamics:
        nodes = (
            GraphNode(HomeostasisVariable.ENERGY.value, equilibrium=0.7, recovery_rate=0.08),
            GraphNode(HomeostasisVariable.STRESS.value, equilibrium=0.2, recovery_rate=0.12),
            GraphNode(HomeostasisVariable.CURIOSITY.value, equilibrium=0.5, recovery_rate=0.06),
            GraphNode(HomeostasisVariable.COGNITIVE_LOAD.value, equilibrium=0.3, recovery_rate=0.10),
            GraphNode(HomeostasisVariable.SOCIAL_NEED.value, equilibrium=0.4, recovery_rate=0.05),
            GraphNode(HomeostasisVariable.METASTABILITY.value, equilibrium=0.3, recovery_rate=0.20),
        )
        edges = (
            GraphEdge(
                source=HomeostasisVariable.STRESS.value,
                destination=HomeostasisVariable.ENERGY.value,
                weight=-0.18,
                transfer=SigmoidTransfer(),
                adaptive_rule=CorrelationAdaptiveRule(rate=0.004),
            ),
            GraphEdge(
                source=HomeostasisVariable.COGNITIVE_LOAD.value,
                destination=HomeostasisVariable.STRESS.value,
                weight=0.22,
                transfer=HillTransfer(),
                adaptive_rule=CorrelationAdaptiveRule(rate=0.006),
            ),
            GraphEdge(
                source=HomeostasisVariable.ENERGY.value,
                destination=HomeostasisVariable.CURIOSITY.value,
                weight=0.14,
                transfer=ExponentialTransfer(),
                adaptive_rule=StaticAdaptiveRule(),
            ),
            GraphEdge(
                source=HomeostasisVariable.CURIOSITY.value,
                destination=HomeostasisVariable.COGNITIVE_LOAD.value,
                weight=0.10,
                transfer=TanhTransfer(gain=1.5),
                adaptive_rule=CorrelationAdaptiveRule(rate=0.003),
            ),
            GraphEdge(
                source=HomeostasisVariable.SOCIAL_NEED.value,
                destination=HomeostasisVariable.STRESS.value,
                weight=0.08,
                transfer=SigmoidTransfer(),
                adaptive_rule=StaticAdaptiveRule(),
            ),
            GraphEdge(
                source=HomeostasisVariable.STRESS.value,
                destination=HomeostasisVariable.COGNITIVE_LOAD.value,
                weight=0.12,
                transfer=ExponentialTransfer(),
                adaptive_rule=CorrelationAdaptiveRule(rate=0.004),
            ),
        )
        return GraphDynamics(nodes=nodes, edges=edges)
