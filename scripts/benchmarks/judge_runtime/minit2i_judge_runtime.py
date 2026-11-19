"""vLLM general plugin for the VLM judges; score_judge.py puts this directory on the judge's PYTHONPATH.

vllm 0.24.0 prefers its FlashAttention-4 kernels (CuTe DSL 4.5.2) on compute capability 10.x; on 10.3 (e.g. B300)
those kernels do not build. Both judges' vision encoders take that path, so on 10.3 this plugin selects vLLM's
FlashAttention-2 backend. The language models run on FlashInfer either way.
"""
REASON = 'compute capability 10.3: the FlashAttention-4 kernels of vllm 0.24.0 (CuTe DSL 4.5.2) abort while compiling'


def register():
    import logging
    from vllm.platforms import current_platform
    from vllm.vllm_flash_attn import flash_attn_interface
    capability = current_platform.get_device_capability() if current_platform.is_cuda() else None
    if capability is not None and (capability.major, capability.minor) == (10, 3) and flash_attn_interface.FA4_AVAILABLE:
        flash_attn_interface.FA4_AVAILABLE = False
        flash_attn_interface.FA4_UNAVAILABLE_REASON = REASON
        logging.getLogger('vllm').warning('minit2i_judge_runtime: FlashAttention-4 disabled (%s); using FlashAttention-2', REASON)
