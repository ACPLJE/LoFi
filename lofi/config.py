"""Configuration for LoFi KV cache compression.

Every knob is an explicit field here and an explicit command-line flag in
:mod:`scripts.run_longbench`, so a run is fully described by its arguments.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

MERGE_MODES = ("cp", "evict", "uniform")
MASS_MODES = ("logn", "none")
VALUE_MODES = ("mass", "imp_mass", "adaptive")
SIM_MODES = ("low", "full")
RESCALE_MODES = ("h", "full", "none")


@dataclass
class LoFiConfig:
    """Compression settings for one model.

    Attributes:
        hh_ratio: Fraction of the prefill kept as the importance-selected
            (heavy-hitter) region.
        recent_ratio: Fraction of the prefill kept as the recent window.
            The retained cache target is ``hh_ratio + recent_ratio``.
        merge_mode: ``cp`` is the content-preserving merge (low-frequency
            channels of the target are left untouched, high-frequency channels
            are blended with the source). ``uniform`` is a naive 50:50 merge
            over all channels. ``evict`` drops the source outright, which
            isolates the selection criterion from the merge operator.
        mass_mode: ``logn`` adds log(merge count) to the decode attention
            logits, restoring the attention mass of the tokens folded into
            each survivor. ``none`` disables the correction.
        value_mode: How values are combined. ``mass`` is the count-weighted
            average, which on the first pass reduces to a plain 50:50 mean.
            ``imp_mass`` additionally weights by attention importance;
            ``adaptive`` uses a similarity-dependent blend instead.
        sim_mode: Channels used to score candidate pairs: ``low`` restricts
            cosine similarity to the position-invariant channels (LoFi-Sel),
            ``full`` uses all channels.
        key_rescale: Norm restoration after a key merge. ``h`` rescales only
            the high-frequency block, which keeps the low-frequency content
            bit-exact; ``full`` rescales the whole vector; ``none`` skips it.
        protect_ratio: Fraction of the compressible region, highest attention
            mass first, that is never merged away.
        max_passes: Cap on merge passes. Each pass pairs consecutive members of
            the active set and alternates pair parity, so several passes are
            needed to reach budgets below 50% retention.
        importance_accumulate: Add the source token's importance to the target
            after a merge, so a survivor representing many tokens ranks higher
            in later passes.
        disabled: Skip compression entirely. Useful as a full-cache reference
            that still runs through the same attention kernels.
        debug: Print per-layer retention statistics once per sequence.
    """

    hh_ratio: float = 0.1
    recent_ratio: float = 0.1
    merge_mode: str = "cp"
    mass_mode: str = "logn"
    value_mode: str = "mass"
    sim_mode: str = "low"
    key_rescale: str = "h"
    protect_ratio: float = 0.1
    max_passes: int = 24
    importance_accumulate: bool = True
    disabled: bool = False
    debug: bool = False

    def __post_init__(self) -> None:
        for name, allowed in (
            ("merge_mode", MERGE_MODES),
            ("mass_mode", MASS_MODES),
            ("value_mode", VALUE_MODES),
            ("sim_mode", SIM_MODES),
            ("key_rescale", RESCALE_MODES),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name}={value!r} not in {allowed}")
        if not 0.0 <= self.protect_ratio < 1.0:
            raise ValueError(f"protect_ratio must be in [0, 1), got {self.protect_ratio}")
        if self.max_passes < 1:
            raise ValueError(f"max_passes must be >= 1, got {self.max_passes}")
        for name in ("hh_ratio", "recent_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def target_retention(self) -> float:
        """Fraction of the prefill the compressed cache should retain."""
        return self.hh_ratio + self.recent_ratio

    def to_dict(self) -> dict[str, Any]:
        """Flat dict of every field, for run manifests."""
        return asdict(self)


#: Named settings referenced by the ablation tables. ``main`` is the method.
PRESETS: dict[str, dict[str, Any]] = {
    # LoFi: low-frequency selection + content-preserving merge + mass correction.
    "main": {},
    # Mass correction removed; isolates its contribution.
    "no_mass": {"mass_mode": "none", "value_mode": "adaptive"},
    # LoFi selection, but the source token is evicted instead of merged.
    # Isolates the selection criterion from the merge operator.
    "select_evict": {"merge_mode": "evict", "mass_mode": "none"},
    # Naive 50:50 full-channel merge, keeping LoFi selection.
    "uniform_merge": {"merge_mode": "uniform"},
    # LoFi merge, but candidates scored on all channels instead of low-frequency.
    "full_sim": {"sim_mode": "full"},
    # No compression; full-cache reference through the same kernels.
    "full_cache": {"disabled": True},
}


def add_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Register the LoFi flags on `parser`. Returns the argument group."""
    group = parser.add_argument_group("LoFi compression")
    defaults = LoFiConfig()
    group.add_argument("--preset", choices=sorted(PRESETS), default="main",
                       help="named setting applied before the individual flags below")
    group.add_argument("--hh-ratio", type=float, default=defaults.hh_ratio)
    group.add_argument("--recent-ratio", type=float, default=defaults.recent_ratio)
    group.add_argument("--merge-mode", choices=MERGE_MODES, default=None)
    group.add_argument("--mass-mode", choices=MASS_MODES, default=None)
    group.add_argument("--value-mode", choices=VALUE_MODES, default=None)
    group.add_argument("--sim-mode", choices=SIM_MODES, default=None)
    group.add_argument("--key-rescale", choices=RESCALE_MODES, default=None)
    group.add_argument("--protect-ratio", type=float, default=None)
    group.add_argument("--max-passes", type=int, default=None)
    group.add_argument("--no-importance-accumulate", dest="importance_accumulate",
                       action="store_false", default=None)
    group.add_argument("--disable-compression", dest="disabled",
                       action="store_true", default=None)
    group.add_argument("--debug-retention", dest="debug",
                       action="store_true", default=None)
    return group


def from_args(args: argparse.Namespace) -> LoFiConfig:
    """Build a config from parsed arguments: preset first, explicit flags win."""
    values: dict[str, Any] = dict(PRESETS[args.preset])
    values["hh_ratio"] = args.hh_ratio
    values["recent_ratio"] = args.recent_ratio
    for name in ("merge_mode", "mass_mode", "value_mode", "sim_mode", "key_rescale",
                 "protect_ratio", "max_passes", "importance_accumulate",
                 "disabled", "debug"):
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override
    return LoFiConfig(**values)
