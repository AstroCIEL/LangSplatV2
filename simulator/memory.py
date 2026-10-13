"""External DRAM bus and bank-level PIM timing.

Time unit of the whole simulator is one ASIC cycle. At 1 GHz that is 1 ns,
so the LPDDR5 parameters quoted in nanoseconds are already cycle counts.

External transfers share one aggregate bus (51.2 GB/s across 4 channels).
PIM reductions do not cross that bus: an SPU reads its bank row buffer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import simpy

from simulator.structures import merge_intervals


@dataclass
class DramConfig:
    memory_bandwidth_gbps: float = 51.2
    clock_frequency_ghz: float = 1.0
    read_latency_hiding_rate: float = 0.3
    row_buffer_bytes: int = 2048
    num_channels: int = 4
    tRCD_ns: float = 18.0
    tRP_ns: float = 18.0
    tRAS_ns: float = 42.0
    tWR_ns: float = 34.0
    # Not listed in the paper; standard LPDDR5 companions so that bank-group
    # command spacing and refresh are actually charged.
    tRRD_ns: float = 5.0
    tREFI_ns: float = 3906.0
    tRFC_ns: float = 180.0


@dataclass
class PimConfig:
    placement: str = "pim"          # "pim" | "asic"
    pim_frequency_ghz: float = 0.2
    asic_frequency_ghz: float = 1.0
    num_channels: int = 4
    banks_per_channel: int = 16
    spus_per_channel: int = 8       # one SPU per two banks → 32 SPUs
    lanes: int = 8                  # FP16 lanes; the second dot product is concurrent
    clip_dim: int = 512
    feature_bytes: int = 1024       # 512 x FP16
    result_bytes_coarse: float = 4  # (s, n) as two FP16
    result_bytes_fine: float = 0.125  # one predicate bit
    row_buffer_bytes: int = 2048
    pdu_cycles_per_pixel: float = 1.0  # PIM-domain cycles, fine stage only
    num_queries: float = 1.0           # text prompts matched against one feature map
    tRCD_ns: float = 18.0
    tRP_ns: float = 18.0
    tRAS_ns: float = 42.0
    tRRD_ns: float = 5.0
    tREFI_ns: float = 3906.0
    tRFC_ns: float = 180.0

    @property
    def num_spus(self) -> int:
        return self.num_channels * self.spus_per_channel

    @property
    def asic_per_pim_cycle(self) -> float:
        if self.pim_frequency_ghz <= 0:
            return 1.0
        return self.asic_frequency_ghz / self.pim_frequency_ghz


class DramSystem:
    """One shared external bus. Bytes are counted even when latency is hidden."""

    def __init__(self, env: simpy.Environment, config: DramConfig):
        self.env = env
        self.config = config
        self.bus = simpy.Resource(env, capacity=1)
        self.bytes: Dict[str, float] = {}
        self._intervals: List[Tuple[float, float]] = []
        self.bytes_per_cycle = (
            config.memory_bandwidth_gbps * 1e9 / (config.clock_frequency_ghz * 1e9)
        )

    def transfer(self, nbytes: float, category: str, hide: bool = False):
        return self.env.process(self._transfer(nbytes, category, hide))

    def _transfer(self, nbytes: float, category: str, hide: bool):
        if nbytes <= 0:
            return
        self.bytes[category] = self.bytes.get(category, 0.0) + nbytes
        cycles = self.transfer_cycles(nbytes, category, hide)
        if cycles <= 0:
            return
        with self.bus.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(cycles)
            self._intervals.append((start, self.env.now))

    def transfer_cycles(self, nbytes: float, category: str, hide: bool) -> float:
        cfg = self.config
        if self.bytes_per_cycle <= 0 or nbytes <= 0:
            return 0.0
        bw_cycles = nbytes / self.bytes_per_cycle
        nrows = math.ceil(nbytes / cfg.row_buffer_bytes) if cfg.row_buffer_bytes > 0 else 0
        # ACT commands on a channel are spaced by tRRD. One tRCD opens the first row.
        timing = nrows * cfg.tRRD_ns + cfg.tRCD_ns
        if category == "feature_write":
            timing += cfg.tWR_ns
        refresh = bw_cycles * (cfg.tRFC_ns / cfg.tREFI_ns) if cfg.tREFI_ns > 0 else 0.0
        if hide:
            visible = bw_cycles * (1.0 - cfg.read_latency_hiding_rate)
            return max(0.0, visible) + refresh
        return max(bw_cycles, timing) + refresh

    def busy_cycles(self) -> float:
        return merge_intervals(self._intervals)


def pim_compute_cycles(n_pixels: float, cfg: PimConfig, stage: str) -> float:
    """ASIC cycles for matching ``num_queries`` prompts against ``n_pixels`` vectors.

    The feature map is decoded once. Each query is one more \(s=f\\cdot q\) pass.
    \(n=f\\cdot f\) shares that pass, so it does not add a second stream.
    A row is opened once and kept open for every query on its pixels, then closed.
    Pixels are striped across banks. Bank-group ACTs on a channel are spaced by
    tRRD, and refresh adds a tRFC/tREFI tax. ASIC-side SPUs use the same
    datapath and 200 MHz clock; only the external feature readback differs.
    """
    queries = float(cfg.num_queries)
    if n_pixels <= 0 or cfg.num_spus <= 0 or queries <= 0:
        return 0.0
    per_spu = math.ceil(n_pixels / cfg.num_spus)
    pixel_bytes = max(1, cfg.feature_bytes)
    pix_per_row = max(1, cfg.row_buffer_bytes // pixel_bytes)
    rows = math.ceil(per_spu / pix_per_row)
    stream_per_pix = math.ceil(cfg.clip_dim / max(1, cfg.lanes)) * cfg.asic_per_pim_cycle
    row_body = max(cfg.tRAS_ns, pix_per_row * stream_per_pix * queries)
    row_cycle = cfg.tRCD_ns + row_body + cfg.tRP_ns
    spus_per_ch = max(1, cfg.num_spus // max(1, cfg.num_channels))
    bank_group = (spus_per_ch - 1) * cfg.tRRD_ns
    compute = rows * (row_cycle + bank_group)
    refresh = compute * (cfg.tRFC_ns / cfg.tREFI_ns) if cfg.tREFI_ns > 0 else 0.0
    pdu = 0.0
    if stage == "fine" and cfg.pdu_cycles_per_pixel > 0:
        # One PDU per channel. Each query needs its own predicate.
        per_pdu = math.ceil(n_pixels / max(1, cfg.num_channels))
        pdu = per_pdu * cfg.pdu_cycles_per_pixel * cfg.asic_per_pim_cycle * queries
    return compute + refresh + pdu


def language_payload_bytes(sde_cfg: dict) -> int:
    """Bytes of the latent carried on each Gaussian.

    Sparse mode stores ``decode_k`` FP16 weights and uint16 indices per level,
    which is the quick-render layout. Dense mode stores one FP16 per codebook entry.
    """
    levels = int(sde_cfg.get("num_levels", 1))
    if sde_cfg.get("sparse_decode", True):
        terms = int(sde_cfg.get("decode_k", 8))
        return levels * terms * 4
    terms = int(sde_cfg.get("codebook_size", 64))
    return levels * terms * 2
