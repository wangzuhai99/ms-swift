# Copyright (c) Alibaba, Inc. and its affiliates.
"""Model FLOPs Utilization (MFU) statistics for training performance monitoring.

Calculates theoretical FLOPs for transformer models and computes MFU metrics
by comparing achieved throughput against GPU peak performance.
"""

import subprocess
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from swift.utils import get_logger

logger = get_logger()

# bf16 peak TFLOPS for common GPUs
GPU_PEAK_FLOPS = {
    'H800': 989e12,
    'H100': 989e12,
    'A800': 312e12,
    'A100': 312e12,
    'H20': 148e12,
    'L40S': 362e12,
    'L40': 181e12,
    'L20': 119.5e12,
    'A30': 165e12,
    'A10': 125e12,
    'V100': 125e12,
    '4090': 330e12,
    '4080': 206e12,
    '3090': 142e12,
    '910B': 313e12,
}

_DEFAULT_GPU_FLOPS = 142e12


def _sum_if_list(x: Union[int, List[int]]) -> int:
    return sum(x) if isinstance(x, list) else x


@lru_cache(maxsize=1)
def _get_gpu_model() -> str:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return 'Unknown'


@lru_cache(maxsize=1)
def _get_gpu_flops() -> float:
    gpu_name = _get_gpu_model().upper()
    for key, flops in GPU_PEAK_FLOPS.items():
        if key.upper() in gpu_name:
            return flops
    logger.warning(f'Unknown GPU "{_get_gpu_model()}", using default {_DEFAULT_GPU_FLOPS / 1e12:.0f} TFLOPS.')
    return _DEFAULT_GPU_FLOPS


def _extract_model_params(config: Any) -> dict:
    """Extract transformer parameters from a PretrainedConfig or dict-like object."""
    def _get(keys, default=None):
        for k in keys:
            val = getattr(config, k, None) if hasattr(config, '__class__') else config.get(k)
            if val is not None:
                return val
        return default

    hidden_size = _get(['hidden_size', 'n_embd', 'd_model'])
    num_heads = _get(['num_attention_heads', 'n_head', 'num_heads'])
    num_kv_heads = _get(['num_key_value_heads', 'num_kv_heads'], num_heads)
    num_layers = _get(['num_hidden_layers', 'n_layer', 'num_layers'])
    intermediate_size = _get(['intermediate_size', 'n_inner', 'd_ff'])
    vocab_size = _get(['vocab_size'])
    head_dim = _get(['head_dim'])

    if hidden_size is None or num_heads is None or num_layers is None:
        raise ValueError(
            'Cannot extract model params from config. '
            'Missing required fields (hidden_size, num_attention_heads, num_hidden_layers).'
        )

    if head_dim is None:
        head_dim = hidden_size // num_heads
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    return {
        'num_head': num_heads,
        'head_dim': head_dim,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'kv_heads': num_kv_heads,
        'num_layers': num_layers,
        'vocab_size': vocab_size or 0,
    }


def _calculate_decoder_layer_flops(
    num_head: int,
    head_dim: int,
    hidden_size: int,
    intermediate_size: int,
    kv_heads: Optional[int] = None,
    is_causal: bool = False,
    seq_len: Union[int, List[int]] = 1,
    batch_size: int = 1,
    linear_factor: int = 2,
    ffn_factor: int = 2,
) -> Dict:
    """Calculate FLOPs for a single transformer decoder layer.

    Args:
        ffn_factor: Number of independent linear projections in FFN.
            2 for standard FFN (up + down), 3 for GLU variants (gate + up + down).
    """
    if kv_heads is None:
        kv_heads = num_head

    seq_len_per_sample = None if isinstance(seq_len, list) else seq_len // max(batch_size, 1)
    total_seq_len = _sum_if_list(seq_len)

    # QKV projection
    q_flops = linear_factor * total_seq_len * hidden_size * (num_head * head_dim)
    k_flops = linear_factor * total_seq_len * hidden_size * (kv_heads * head_dim)
    v_flops = linear_factor * total_seq_len * hidden_size * (kv_heads * head_dim)

    # Attention scores (quadratic in seq_len per sample)
    if isinstance(seq_len, list):
        attn_scores_flops = sum(
            linear_factor * num_head * s * s * head_dim for s in seq_len
        )
    else:
        attn_scores_flops = (
            linear_factor * num_head * seq_len_per_sample
            * seq_len_per_sample * head_dim * batch_size
        )
    if is_causal:
        attn_scores_flops //= 2

    attn_v_flops = attn_scores_flops

    # Attention output projection
    attn_out_flops = linear_factor * total_seq_len * (num_head * head_dim) * hidden_size

    attention_flops = q_flops + k_flops + v_flops + attn_scores_flops + attn_v_flops + attn_out_flops

    # FFN (gate_proj + up_proj + down_proj for GLU, or just up + down for standard)
    ffn_flops = linear_factor * total_seq_len * hidden_size * intermediate_size * ffn_factor

    total_flops = attention_flops + ffn_flops
    return {
        'total_flops': total_flops,
        'attention_flops': attention_flops,
        'ffn_flops': ffn_flops,
    }


def _calculate_model_flops(
    model_params: dict,
    seq_len: Union[int, List[int]] = 1,
    batch_size: int = 1,
    is_causal: bool = True,
    include_lm_head: bool = True,
    ffn_factor: int = 3,
) -> Dict:
    """Calculate total forward-pass FLOPs for a transformer model."""
    linear_factor = 2

    layer_flops = _calculate_decoder_layer_flops(
        num_head=model_params['num_head'],
        head_dim=model_params['head_dim'],
        hidden_size=model_params['hidden_size'],
        intermediate_size=model_params['intermediate_size'],
        kv_heads=model_params.get('kv_heads'),
        is_causal=is_causal,
        seq_len=seq_len,
        batch_size=batch_size,
        linear_factor=linear_factor,
        ffn_factor=ffn_factor,
    )

    total_flops = layer_flops['total_flops'] * model_params['num_layers']

    result = {
        'total_flops': total_flops,
        'per_layer_flops': layer_flops['total_flops'],
        'num_layers': model_params['num_layers'],
    }

    if include_lm_head and model_params.get('vocab_size', 0) > 0:
        lm_head_flops = linear_factor * _sum_if_list(seq_len) * model_params['hidden_size'] * model_params['vocab_size']
        result['total_flops'] += lm_head_flops
        result['lm_head_flops'] = lm_head_flops

    return result


class MFUStats:
    """Tracks token counts and computes MFU during training.

    Directly accepts a HuggingFace PretrainedConfig (model.config),
    no need to manually pass config paths or pre-extracted dicts.

    Args:
        config: HuggingFace PretrainedConfig object (i.e. model.config).
        logging_steps: Number of training steps per logging interval.
        include_lm_head: Whether to include LM head in FLOPs calculation.
        is_causal: Whether the model uses causal attention.
        ffn_factor: FFN linear projection count (3 for GLU, 2 for standard).
    """

    def __init__(
        self,
        config: Any,
        logging_steps: int = 5,
        include_lm_head: bool = True,
        is_causal: bool = True,
        ffn_factor: int = 3,
    ):
        self._model_params = _extract_model_params(config)
        self._logging_steps = logging_steps
        self._include_lm_head = include_lm_head
        self._is_causal = is_causal
        self._ffn_factor = ffn_factor

        self._step_tokens = defaultdict(int)
        self._cumulative = defaultdict(float)

    def set(self, num_tokens: int, num_samples: int) -> None:
        """Accumulate token and sample counts (called every training step)."""
        self._step_tokens['num_tokens'] += int(num_tokens)
        self._step_tokens['num_samples'] += int(num_samples)

    def compute(self, elapsed_secs: float, global_step: int) -> Dict[str, float]:
        """Compute MFU metrics for the current logging interval.

        Args:
            elapsed_secs: Wall-clock seconds since last log.
            global_step: Current global training step.

        Returns:
            Dict of MFU metrics suitable for logging.
        """
        tokens = self._step_tokens
        steps = self._logging_steps

        if tokens['num_tokens'] == 0 or steps == 0 or elapsed_secs <= 0:
            self._step_tokens = defaultdict(int)
            return {}

        avg_tokens_per_step = round(tokens['num_tokens'] / steps)
        avg_samples_per_step = max(round(tokens['num_samples'] / steps), 1)
        secs_per_step = elapsed_secs / steps

        result = self._calc_mfu(
            total_seq_len=avg_tokens_per_step,
            batch_size=avg_samples_per_step,
            secs_per_step=secs_per_step,
        )

        mfu_value = result.get('mfu', 0.0)
        tflops = result.get('training_flops_tflops', 0.0)

        self._cumulative['mfu'] += mfu_value * steps
        self._cumulative['tflops'] += tflops * steps

        logs = {
            'perf/mfu': round(mfu_value, 6),
            'perf/tflops_per_gpu': round(tflops, 4),
        }
        if global_step > 0:
            logs['perf/mfu_avg'] = round(self._cumulative['mfu'] / global_step, 6)

        self._step_tokens = defaultdict(int)
        return logs

    def _calc_mfu(self, total_seq_len: int, batch_size: int, secs_per_step: float) -> Dict:
        flops = _calculate_model_flops(
            self._model_params,
            seq_len=total_seq_len,
            batch_size=batch_size,
            is_causal=self._is_causal,
            include_lm_head=self._include_lm_head,
            ffn_factor=self._ffn_factor,
        )

        gpu_flops = _get_gpu_flops()
        training_flops = flops['total_flops'] * 3
        training_tflops = training_flops / 1e12

        result = {
            'forward_flops': flops['total_flops'],
            'training_flops_tflops': training_tflops,
            'gpu_peak_flops': gpu_flops,
            'flops_per_gpu_ratio': training_flops / gpu_flops,
        }
        if secs_per_step > 0:
            result['mfu'] = result['flops_per_gpu_ratio'] / secs_per_step
        return result
