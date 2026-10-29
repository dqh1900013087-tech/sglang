"""Store already quantized DSA index keys in a packed NPU cache.

Each page contains [page_size * 128 K bytes][page_size FP32 scales].
This module requires Triton Ascend; it does not perform quantization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _set_k_and_s_npu_kernel(
    cache_bytes,
    cache_floats,
    locations,
    key_bytes,
    scales,
    PAGE_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    # Use 64-bit address arithmetic even when locations are supplied as int32.
    location = tl.load(locations + token).to(tl.int64)
    page = location // PAGE_SIZE
    slot = location % PAGE_SIZE
    columns = tl.arange(0, 128)
    key = tl.load(key_bytes + token.to(tl.int64) * 128 + columns)
    scale = tl.load(scales + token)

    key_offset = page * (PAGE_SIZE * 132) + slot * 128 + columns
    # cache_floats is an alias of cache_bytes: offsets are FP32 elements.
    scale_offset = page * (PAGE_SIZE * 33) + PAGE_SIZE * 32 + slot
    tl.store(cache_bytes + key_offset, key)
    tl.store(cache_floats + scale_offset, scale)


def set_index_k_scale_buffer_npu(
    cache: torch.Tensor,
    loc: torch.Tensor,
    index_k: torch.Tensor,
    index_k_scale: torch.Tensor,
    page_size: int,
) -> None:
    """Write K and scale in place with one Triton kernel launch.

    Args:
        cache: Contiguous uint8 [num_pages, page_size * 132] buffer.
        loc: Contiguous int32/int64 [T] physical token slots.
        index_k: Contiguous float8_e4m3fn [T, 128], already quantized by NPU.
        index_k_scale: Contiguous float32 [T] or [T, 1].
        page_size: Tokens per page; 16..1024, divisible by 16, matching
            the packed Ascend 950 quant_lightning_indexer reader.

    All tensors must be on the same NPU. Locations must be unique and in
    [0, num_pages * page_size); padding/duplicate slots must be filtered by
    the caller. These value constraints are not checked here to avoid a
    device synchronization. Inputs must not alias the output cache.

    K is reinterpreted as uint8, not converted: its FP8 bits are preserved.
    Both cache views alias the same allocation. No packing temporary or
    quantization is performed. Unwritten slots are left unchanged.
    """
    if not isinstance(page_size, int) or not (
        16 <= page_size <= 1024 and page_size % 16 == 0
    ):
        raise ValueError("page_size must be a multiple of 16 in [16, 1024]")
    if cache.device.type != "npu":
        raise ValueError("cache must be on an NPU")
    tensors = (cache, loc, index_k, index_k_scale)
    if any(t.device != cache.device for t in tensors):
        raise ValueError("all tensors must be on the same NPU")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    if cache.dtype != torch.uint8 or cache.ndim != 2:
        raise ValueError("cache must be a rank-2 uint8 tensor")
    if cache.shape[0] == 0 or cache.shape[1] != page_size * 132:
        raise ValueError("cache shape must be [num_pages > 0, page_size * 132]")
    if cache.storage_offset() % 4:
        raise ValueError("cache storage offset must be FP32 aligned")
    if loc.ndim != 1 or loc.dtype not in (torch.int32, torch.int64):
        raise ValueError("loc must be a rank-1 int32/int64 tensor")
    num_tokens = loc.numel()
    if index_k.dtype != torch.float8_e4m3fn or index_k.shape != (num_tokens, 128):
        raise ValueError("index_k must be float8_e4m3fn [T, 128]")
    if index_k_scale.dtype != torch.float32 or index_k_scale.shape not in (
        (num_tokens,),
        (num_tokens, 1),
    ):
        raise ValueError("index_k_scale must be float32 [T] or [T, 1]")
    if num_tokens == 0:
        return

    _set_k_and_s_npu_kernel[(num_tokens,)](
        cache,
        cache.view(torch.float32),
        loc,
        index_k.view(torch.uint8),
        index_k_scale,
        PAGE_SIZE=page_size,
    )
