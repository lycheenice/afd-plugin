# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe AFD rank topology helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from afd_plugin.config import AFDConfig


@dataclass(frozen=True, slots=True)
class AFDRankMapping:
    """Rank mapping for the P2P connector.

    The P2P world always places FFN ranks first, followed by Attention ranks:
    ``[F0, F1, ..., A0, A1, ...]``.

    **Fan-in mode** (``attention_size >= ffn_size``): each FFN rank owns one
    subgroup containing itself at subgroup rank 0 and one or more consecutive
    Attention ranks.  Attention sends to its mapped FFN; the FFN concatenates
    inputs from multiple Attention peers.

    **Fan-out mode** (``attention_size < ffn_size``): each Attention rank owns
    one subgroup containing itself at subgroup rank 0 and one or more
    consecutive FFN ranks.  Attention broadcasts to all FFN peers; only the
    leader FFN (subgroup rank 1) sends the combined result back.
    """

    role: str
    role_rank: int
    world_rank: int
    p2p_rank: int
    attention_size: int
    ffn_size: int
    min_size: int
    ratio: int
    subgroup_index: int
    rank_in_subgroup: int
    subgroup_ranks: tuple[int, ...]
    dp_metadata_destinations: tuple[int, ...] = field(default_factory=tuple)

    # ### PATCH START: AFD fan-out topology (attention < ffn, e.g. 1A2F)
    reversed: bool = False
    # ### PATCH END: AFD fan-out topology

    @property
    def is_attention_top_min_size_rank(self) -> bool:
        return self.ffn_size <= self.world_rank < self.ffn_size + self.min_size

    @property
    def participates_in_dp_metadata_group(self) -> bool:
        return self.world_rank < self.ffn_size or self.is_attention_top_min_size_rank


def topology_from_config(config: AFDConfig) -> tuple[int, int]:
    """Return ``(attention_size, ffn_size)`` for an AFD config."""

    return config.num_attention_ranks, config.num_ffn_ranks


def validate_p2p_topology(config: AFDConfig) -> None:
    attention_size, ffn_size = topology_from_config(config)
    if attention_size == ffn_size:
        return
    if attention_size > ffn_size:
        if attention_size % ffn_size != 0:
            raise ValueError(
                "P2pNcclAFDConnector requires num_attention_ranks to be a "
                "multiple of num_ffn_ranks (fan-in mode), got "
                f"{attention_size} and {ffn_size}",
            )
        return
    # ### PATCH START: AFD fan-out topology (attention < ffn, e.g. 1A2F)
    if ffn_size % attention_size != 0:
        raise ValueError(
            "P2pNcclAFDConnector requires num_ffn_ranks to be a multiple of "
            "num_attention_ranks (fan-out mode), got "
            f"{ffn_size} and {attention_size}",
        )
    # ### PATCH END: AFD fan-out topology


def build_rank_mapping(
    config: AFDConfig,
    role_rank: int | None = None,
) -> AFDRankMapping:
    """Build the P2P rank mapping for one Attention or FFN process.

    Supports both fan-in mode (``attention >= ffn``, the original design where
    one FFN rank serves multiple Attention ranks) and fan-out mode
    (``attention < ffn``, where one Attention rank broadcasts to multiple FFN
    ranks that shard experts via tensor parallelism).
    """

    validate_p2p_topology(config)
    attention_size, ffn_size = topology_from_config(config)
    role_rank = config.afd_role_rank if role_rank is None else role_rank
    if role_rank < 0:
        raise ValueError(f"AFD role rank must be non-negative, got {role_rank}")

    # ### PATCH START: AFD fan-out topology (attention < ffn, e.g. 1A2F)
    reversed = attention_size < ffn_size
    if reversed:
        return _build_rank_mapping_fan_out(
            config, role_rank, attention_size, ffn_size,
        )
    # ### PATCH END: AFD fan-out topology

    if config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        subgroup_index = role_rank // (attention_size // ffn_size)
    elif config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        subgroup_index = role_rank
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    ratio = attention_size // ffn_size
    min_size = min(ffn_size, attention_size)
    ffn_ranks = list(range(ffn_size))
    attention_ranks = list(range(ffn_size, ffn_size + attention_size))
    subgroup_ranks = tuple(
        [ffn_ranks[subgroup_index]]
        + [attention_ranks[subgroup_index * ratio + offset] for offset in range(ratio)],
    )
    rank_in_subgroup = subgroup_ranks.index(world_rank)
    p2p_rank = role_rank + min_size if config.role == "attention" else role_rank

    destinations: list[int] = []
    if ffn_size <= world_rank < ffn_size + min_size:
        local_attention_rank = world_rank - ffn_size
        destination = local_attention_rank
        while destination < ffn_size:
            destinations.append(destination)
            destination += min_size

    return AFDRankMapping(
        role=config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        ratio=ratio,
        subgroup_index=subgroup_index,
        rank_in_subgroup=rank_in_subgroup,
        subgroup_ranks=subgroup_ranks,
        dp_metadata_destinations=tuple(destinations),
    )


# ### PATCH START: AFD fan-out topology (attention < ffn, e.g. 1A2F)
# Patch reason: upstream topology only supports attention >= ffn (fan-in).
# Fan-out mode allows attention < ffn, enabling expert-parallel scaling by
# adding more FFN GPUs. Each attention rank leads a subgroup of ffn_size/attention_size
# FFN ranks.
# Patch functionality: builds reversed subgroups [A_j, F_{j*ratio}, ..., F_{(j+1)*ratio-1}]
# where A is subgroup rank 0 and FFN ranks occupy 1..ratio. DP metadata destinations
# use contiguous blocks instead of stride pattern.
def _build_rank_mapping_fan_out(
    config: AFDConfig,
    role_rank: int,
    attention_size: int,
    ffn_size: int,
) -> AFDRankMapping:
    """Build rank mapping for fan-out mode (attention < ffn).

    World: ``[F0, ..., F_{ffn_size-1}, A0, ..., A_{attention_size-1}]``.
    Each Attention rank ``j`` owns subgroup ``[A_j, F_{j*ratio}, ...,
    F_{(j+1)*ratio-1}]`` with A at subgroup rank 0. Only the first FFN
    (subgroup rank 1) sends the combined result back to Attention.
    """
    ratio = ffn_size // attention_size
    min_size = attention_size

    if config.role == "attention":
        if role_rank >= attention_size:
            raise ValueError(
                "Attention role rank must be within attention size "
                f"(rank={role_rank}, size={attention_size})",
            )
        world_rank = ffn_size + role_rank
        subgroup_index = role_rank
    elif config.role == "ffn":
        if role_rank >= ffn_size:
            raise ValueError(
                "FFN role rank must be within FFN size "
                f"(rank={role_rank}, size={ffn_size})",
            )
        world_rank = role_rank
        subgroup_index = role_rank // ratio
    else:
        raise ValueError(f"unknown AFD role {config.role!r}")

    # Build subgroup: [A_subgroup_index, F_{subgroup_index*ratio}, ..., F_{...+ratio-1}]
    attn_world_rank = ffn_size + subgroup_index
    ffn_world_ranks = [
        subgroup_index * ratio + offset for offset in range(ratio)
    ]
    subgroup_ranks = tuple([attn_world_rank] + ffn_world_ranks)
    rank_in_subgroup = subgroup_ranks.index(world_rank)
    # In fan-out mode all ranks participate in the p2p metadata group.
    p2p_rank = world_rank

    # DP metadata: attention sends to contiguous block of FFN ranks.
    destinations: list[int] = []
    if config.role == "attention":
        destinations = list(range(
            subgroup_index * ratio, (subgroup_index + 1) * ratio,
        ))

    return AFDRankMapping(
        role=config.role,
        role_rank=role_rank,
        world_rank=world_rank,
        p2p_rank=p2p_rank,
        attention_size=attention_size,
        ffn_size=ffn_size,
        min_size=min_size,
        ratio=ratio,
        subgroup_index=subgroup_index,
        rank_in_subgroup=rank_in_subgroup,
        subgroup_ranks=subgroup_ranks,
        dp_metadata_destinations=tuple(destinations),
        reversed=True,
    )
# ### PATCH END: AFD fan-out topology


__all__ = [
    "AFDRankMapping",
    "build_rank_mapping",
    "topology_from_config",
    "validate_p2p_topology",
]
