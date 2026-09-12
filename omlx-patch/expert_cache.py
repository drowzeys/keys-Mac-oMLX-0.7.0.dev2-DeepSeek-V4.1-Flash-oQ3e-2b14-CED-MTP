# SPDX-License-Identifier: Apache-2.0
"""SSD-paged routed experts for DeepSeek-V4.1-Flash.

Engram offload frees the n-gram tables, but oQ3e's routed-expert weights are still
220 GiB, which does not fit a 256 GB machine once KV and activations are accounted
for. This pages whole MoE layers' expert tables to SSD and keeps a bounded hot-expert
cache resident, so a token reads only the experts it actually routes to.

Whole-layer granularity is deliberate. Routing picks ``num_experts_per_tok`` of
``n_routed_experts`` per layer, so paging N layers costs ``N * k`` expert reads per
token; a global LRU of the same total size instead pays misses across all 40 layers.

Each call gathers the unique routed experts into a compact table in sorted order and
remaps the indices into it, so ``sorted_indices`` still holds and
``deepseek_affine_gather_qmm_blocks`` stays eligible. Misses read through
``TensorFile.read(key, rows=...)``, whose advanced indexing on dim 0 is exactly one
expert's slice of a stacked ``[experts, out, in]`` tensor.
"""

import logging
import os
import threading

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .quantization import QuantizedProjection
from .storage import decode_array

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_GB = 8.0


class ExpertStore:
    """Resident hot-expert slots for one projection, over an mmap'd checkpoint shard.

    Held as a plain attribute of the projection so ``nn.Module`` parameter discovery
    never walks these arrays: a paged projection must contribute nothing to
    ``model.parameters()``, exactly as ``DiskEngramEmbedding`` contributes nothing.
    """

    def __init__(self, reader, name, keys, experts, capacity, expert_bytes,
                 group_size=64):
        self.reader = reader
        self.name = name
        self.keys = keys                  # (weight_key, scales_key, biases_key|None)
        self.experts = experts
        self.capacity = max(1, min(capacity, experts))
        self.expert_bytes = expert_bytes
        self.group_size = group_size
        self.lock = threading.RLock()
        self.slots = {}                   # expert id -> slot
        self.recent = []                  # slots, least-recently-used first
        self.arrays = None
        self.hits = self.misses = self.bytes_read = self.bypassed = 0
        self.input_dims = None

    def _install(self, slot, rows, position):
        for index, key in enumerate(self.keys):
            if key is None:
                continue
            raw, dtype = rows[index]
            value = decode_array(raw[position], dtype)
            if self.arrays is None or self.arrays[index] is None:
                self._allocate(index, value)
            self.arrays[index][slot] = value

    def _allocate(self, index, sample):
        if self.arrays is None:
            self.arrays = [None, None, None]
        array = mx.zeros((self.capacity, *sample.shape), dtype=sample.dtype)
        mx.eval(array)
        self.arrays[index] = array

    def gather(self, wanted):
        """Compacted tables for ``wanted`` (sorted expert ids), cached when it fits.

        A routing step that needs more distinct experts than the cache holds — large
        batches fan out toward the whole table — bypasses the cache and reads exactly
        those rows. Caching them would thrash every slot for no reuse.
        """
        if len(wanted) > self.capacity:
            return self._read_direct(wanted)
        slots = self.acquire(wanted)
        take = mx.array(np.asarray(slots, dtype=np.uint32))
        return tuple(
            None if array is None else mx.take(array, take, axis=0)
            for array in self.arrays
        )

    def _read_direct(self, wanted):
        rows = [
            self.reader.read(key, rows=wanted) if key else None for key in self.keys
        ]
        self.misses += len(wanted)
        self.bypassed += len(wanted)
        self.bytes_read += len(wanted) * self.expert_bytes
        values = tuple(
            None if row is None else decode_array(row[0], row[1]) for row in rows
        )
        mx.eval([value for value in values if value is not None])
        if self.input_dims is None and values[1] is not None:
            self.input_dims = values[1].shape[-1] * self.group_size
        return values

    def acquire(self, wanted):
        """Make every expert in ``wanted`` resident; return one slot per expert."""
        with self.lock:
            missing = [expert for expert in wanted if expert not in self.slots]
            if missing:
                rows = [
                    self.reader.read(key, rows=missing) if key else None
                    for key in self.keys
                ]
                for position, expert in enumerate(missing):
                    slot = self._free_slot(wanted)
                    self._install(slot, rows, position)
                    self.slots[expert] = slot
                    self.recent.append(slot)
                self.bytes_read += len(missing) * self.expert_bytes
                self.misses += len(missing)
                mx.eval([a for a in self.arrays if a is not None])
            self.hits += len(wanted) - len(missing)
            for expert in wanted:
                slot = self.slots[expert]
                if self.recent and self.recent[-1] != slot:
                    if slot in self.recent:
                        self.recent.remove(slot)
                    self.recent.append(slot)
            if self.input_dims is None and self.arrays and self.arrays[1] is not None:
                self.input_dims = self.arrays[1].shape[-1] * self.group_size
            return [self.slots[expert] for expert in wanted]

    def _free_slot(self, protected):
        if len(self.slots) < self.capacity:
            return len(self.slots)
        protected = set(protected)
        occupant = {slot: expert for expert, slot in self.slots.items()}
        for slot in list(self.recent):
            expert = occupant.get(slot)
            if expert is None or expert in protected:
                continue
            del self.slots[expert]
            self.recent.remove(slot)
            return slot
        raise RuntimeError(
            f"{self.name}: hot-expert cache holds {self.capacity} slots but one routing "
            f"step needs {len(protected)}; raise OMLX_DSV41_EXPERT_CACHE_GB"
        )

    def stats(self):
        total = self.hits + self.misses
        return {
            "experts": self.experts,
            "capacity": self.capacity,
            "resident": len(self.slots),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "bypassed": self.bypassed,
            "bytes_read": self.bytes_read,
        }

    def close(self):
        self.arrays = None
        self.slots.clear()
        self.recent.clear()


class PagedExpertProjection(QuantizedProjection):
    """Routed-expert projection whose stacked weights live on SSD."""

    def __init__(self, store, spec):
        nn.Module.__init__(self)
        self._store = store
        self.bits = spec["bits"]
        self.mode = spec["mode"]
        self.group_size = spec.get("group_size", 32)
        self.quantize_input = spec.get("quantize_input", True)

    @property
    def input_dims(self):
        return self._store.input_dims

    def project_quantized(self, x, indices=None, sorted_indices=False, block_plan=None):
        if indices is None:
            raise ValueError("Paged routed experts require routing indices")
        routed = np.asarray(indices, copy=False)
        wanted = np.unique(routed)                     # np.unique returns sorted values
        # Compacted in routing order, so remapped indices stay sorted and the native
        # affine block kernel remains eligible.
        weight, scales, biases = self._store.gather([int(e) for e in wanted])
        self.weight, self.scales = weight, scales
        if biases is not None:
            self.biases = biases
        lookup = np.full(self._store.experts, -1, dtype=np.int32)
        lookup[wanted] = np.arange(wanted.size, dtype=np.int32)
        remapped = mx.array(lookup[routed])
        try:
            return QuantizedProjection.project_quantized(
                self, x, remapped, sorted_indices=sorted_indices, block_plan=block_plan
            )
        finally:
            # Drop the compacted view so it is not mistaken for a resident parameter.
            for field in ("weight", "scales", "biases"):
                if field in self:
                    delattr(self, field)


def expert_bytes_per_layer(config, spec_bits, group_size=64):
    """Packed bytes for one expert's w1+w3+w2 at the checkpoint's expert precision."""
    # ModelConfig field names are the aliased ones: hidden_size -> dim,
    # moe_intermediate_size -> moe_inter_dim (see config.ModelConfig.from_dict).
    hidden, inter = config.dim, config.moe_inter_dim
    elements = 2 * inter * hidden + hidden * inter
    scales = elements // group_size
    return elements * spec_bits // 8 + scales * 2 * 2


def configured_capacity(experts, expert_bytes, layers):
    """Hot-expert slots per layer from ``OMLX_DSV41_EXPERT_CACHE_GB`` (total budget)."""
    budget = os.environ.get("OMLX_DSV41_EXPERT_CACHE_GB")
    gigabytes = float(budget) if budget else _DEFAULT_CACHE_GB
    if expert_bytes <= 0 or not layers:
        return experts
    per_layer = gigabytes * 1024**3 / max(1, layers)
    return max(2, min(experts, int(per_layer // expert_bytes)))


def choose_paged_layers(moe_layers, resident_bytes, ceiling_bytes, layer_bytes):
    """Fewest whole MoE layers whose eviction brings residency under the ceiling.

    Deepest layers go first: a cold read on layer 0 would stall every prefill.
    """
    if ceiling_bytes <= 0 or layer_bytes <= 0 or resident_bytes <= ceiling_bytes:
        return []
    needed = resident_bytes - ceiling_bytes
    count = min(int(-(-needed // layer_bytes)), len(moe_layers))
    return sorted(sorted(moe_layers, reverse=True)[:count])


def close_paged_experts(model):
    """Release paged-expert mmap handles. Called from ``Model.close``.

    These readers must outlive ``load()`` — a ``PagedExpertProjection`` reads through
    them on every routing step — so they are owned by the model, not by loading's
    ExitStack.
    """
    for reader in getattr(model, "_paged_expert_readers", ()) or ():
        try:
            reader.close()
        except Exception:  # a half-open mapping must not mask the real teardown error
            logger.debug("paged expert reader close failed", exc_info=True)
    model._paged_expert_readers = []


def expert_cache_stats(model):
    """Per-projection paging counters, for benchmarks and regression gates."""
    found = {}

    def walk(module, prefix):
        if isinstance(module, PagedExpertProjection):
            found[prefix.rstrip(".")] = module._store.stats()
            return
        children = getattr(module, "children", None)
        if children is None:
            return
        for name, child in children().items():
            walk(child, f"{prefix}{name}.")

    walk(model, "")
    return found
