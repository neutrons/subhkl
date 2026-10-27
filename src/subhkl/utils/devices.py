"""Which devices a run may use, and how a batch axis spreads across them.

Multi-GPU is strictly opt-in.  JAX's CUDA client claims memory on every
visible device the moment the backend initializes, so a finder that silently
used all of them would make one benchmark process the neighbour that OOMs
every other job on the machine.  The contract is therefore:

* Without ``--multi-gpu`` a command restricts itself to one device (the first
  visible one), and behaves exactly as it always has.
* With ``--multi-gpu`` it builds a mesh over every visible device and shards
  one outer, embarrassingly parallel batch axis across it: images in the
  finder's solve, independent restarts in the indexer.  Which devices are
  visible stays the operator's decision, via ``CUDA_VISIBLE_DEVICES``.

Sharding only the outer axis keeps the partitioning trivial for XLA: every
per-item computation is independent, so the compiler never needs to invent a
cross-device decomposition of the solver itself, only to run disjoint slices
of the batch on different devices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def restrict_to_first_device() -> None:
    """Best effort: keep the CUDA client from claiming every visible GPU.

    Indexes into the ``CUDA_VISIBLE_DEVICES`` set, so an operator who pinned a
    process to one card keeps exactly that card.  Only effective before the
    JAX backend initializes; once arrays exist the visibility is settled, and
    this quietly does nothing rather than fail a run over memory that has
    already been claimed.
    """
    try:
        jax.config.update("jax_cuda_visible_devices", "0")
    except Exception:
        pass


def batch_devices(multi_gpu: bool) -> list:
    """The devices one batched computation may spread over."""
    devices = list(jax.devices())
    return devices if multi_gpu else devices[:1]


def batch_sharding(devices: list):
    """A sharding that splits an array's leading axis across ``devices``."""
    mesh = jax.sharding.Mesh(np.asarray(devices), ("batch",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))


def pad_to_multiple(array, multiple: int):
    """Pad the leading axis up to a multiple by repeating the final element.

    Repeating a real element rather than appending zeros keeps the padding
    inside the distribution the computation was compiled for -- a zero image,
    for example, would hand the Poisson solver a zero background.  Callers
    discard the padded tail, so the only cost is redundant work on at most
    ``multiple - 1`` items.
    """
    n = int(array.shape[0])
    pad = (-n) % int(multiple)
    if pad == 0:
        return array
    return jnp.concatenate([array, jnp.repeat(array[-1:], pad, axis=0)], axis=0)
