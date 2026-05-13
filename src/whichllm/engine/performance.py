"""Token generation speed estimation."""

from __future__ import annotations

from whichllm.engine.quantization import estimate_weight_bytes
from whichllm.engine.quantization import effective_quant_type
from whichllm.hardware.types import GPUInfo
from whichllm.models.types import GGUFVariant, ModelInfo


# Per-quant efficiency factors applied to the theoretical bandwidth-bound
# tok/s. These reflect empirical llama.cpp / vLLM measurements: 4-bit GGUFs
# achieve the highest fraction of memory-bandwidth-limited theoretical
# throughput because the dequantization kernel is fast and weight reads
# dominate; 8-bit and FP16 drop because more compute is required per byte.
_QUANT_EFFICIENCY: dict[str, float] = {
    "F32":     0.30,
    "F16":     0.40,
    "BF16":    0.40,
    "Q8_0":    0.45,
    "Q6_K":    0.50,
    "Q5_K_M":  0.52,
    "Q5_K_S":  0.52,
    "Q5_0":    0.50,
    "Q4_K_M":  0.55,
    "Q4_K_S":  0.55,
    "Q4_0":    0.53,
    "Q3_K_M":  0.50,
    "Q3_K_S":  0.48,
    "Q3_K_L":  0.50,
    "Q2_K":    0.45,
    "IQ4_XS":  0.52,
    "IQ4_NL":  0.50,
    "IQ3_S":   0.45,
    "IQ3_M":   0.45,
    "IQ3_XS":  0.45,
    "IQ3_XXS": 0.42,
    "IQ2_S":   0.40,
    "IQ2_M":   0.40,
    "IQ2_XXS": 0.38,
    "IQ1_M":   0.35,
    "IQ1_S":   0.35,
    "Q2_0":    0.38,
    "Q1_0":    0.32,
    "TQ2_0":   0.35,
    "TQ1_0":   0.32,
}

_DEFAULT_QUANT_EFFICIENCY = 0.45

# Vendor / backend multiplier applied on top of quant efficiency. CUDA on
# modern data-center GPUs is the reference (1.0); Apple's Metal kernel is
# behind on dequantization; ROCm trails further; older CUDA generations
# also drop.
_BACKEND_FACTOR: dict[str, float] = {
    "nvidia":  1.00,
    "amd":     0.78,
    "apple":   0.82,
    "intel":   0.65,
}


def _backend_factor(gpu: GPUInfo) -> float:
    if gpu.vendor in _BACKEND_FACTOR:
        return _BACKEND_FACTOR[gpu.vendor]
    return 0.7


def _quant_efficiency(model: ModelInfo, variant: GGUFVariant | None) -> float:
    quant = effective_quant_type(model, variant)
    if not quant:
        return _DEFAULT_QUANT_EFFICIENCY
    return _QUANT_EFFICIENCY.get(quant.upper(), _DEFAULT_QUANT_EFFICIENCY)


def estimate_tok_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str = "full_gpu",
) -> float:
    """Estimate tokens per second for inference.

    Model: throughput is bounded by the time it takes to read all weights
    needed per token, multiplied by quant- and backend-specific efficiency
    factors. The default 0.5 efficiency factor used earlier mixed two
    distinct losses (compute kernel quality and offload overhead) into one
    constant — this version separates them so a Q4_K_M model on CUDA scores
    differently from the same model running on Metal or with partial
    offload.
    """
    if gpu is None or fit_type == "cpu_only":
        params_b = model.parameter_count / 1e9
        if model.is_moe and model.parameter_count_active:
            params_b = model.parameter_count_active / 1e9
        if params_b <= 0:
            return 0.0
        # Modern desktop CPUs sustain roughly 4-8 GB/s effective for the
        # bandwidth-bound dequant+matmul loop on a single socket. Quantized
        # 4-bit 7B → ~3.5 GB → ~1-2 tok/s. Approximate with an inverse-size
        # heuristic that gets the right order of magnitude.
        quant_factor = _quant_efficiency(model, variant) / _DEFAULT_QUANT_EFFICIENCY
        return max(0.3, 18.0 / max(params_b, 0.5) * quant_factor)

    model_size = estimate_weight_bytes(model, variant)

    # MoE: only active params need to be read per token. The router itself
    # also touches the shared layers, so we don't drop fully to the active
    # ratio.
    if model.is_moe and model.parameter_count_active:
        active_ratio = model.parameter_count_active / model.parameter_count
        # Floor at 0.25: shared layers and routing always cost some bandwidth.
        active_ratio = max(active_ratio, 0.25)
        effective_read = model_size * active_ratio
    else:
        effective_read = model_size

    bandwidth = gpu.memory_bandwidth_gbps * 1e9 if gpu.memory_bandwidth_gbps else 0
    if bandwidth == 0:
        return 0.0

    theoretical = bandwidth / effective_read

    # Real-world efficiency depends on quant kernel and backend.
    efficiency = _quant_efficiency(model, variant) * _backend_factor(gpu)

    # Partial offload: weight reads that cross the PCIe bus suffer ~10x
    # bandwidth drop; assume 40% of the model lives on CPU and the GPU
    # half completes at full speed.
    if fit_type == "partial_offload":
        efficiency *= 0.45

    return theoretical * efficiency
