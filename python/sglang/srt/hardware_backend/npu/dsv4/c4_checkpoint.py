"""Ascend 950 C4 request-bank / SWA checkpoint bridge."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class C4CheckpointChunk:
    request: int
    begin: int
    end: int
    position: int
    output_begin: int
    output_count: int
    first: bool


def build_checkpoint_chunks(cumulative_lengths, prefixes, page_size):
    """Split eager prefill before live state advances past a page boundary."""
    chunks = []
    output_begin = 0
    for request, (lo, hi) in enumerate(zip(cumulative_lengths, cumulative_lengths[1:])):
        position = int(prefixes[request])
        first = True
        while lo < hi:
            count = min(hi - lo, page_size - position % page_size)
            outputs = (position + count) // 4 - position // 4
            chunks.append(
                C4CheckpointChunk(
                    request, lo, lo + count, position, output_begin, outputs, first
                )
            )
            lo += count
            position += count
            output_begin += outputs
            first = False
    return chunks


def _swa_pages(backend, requests, positions):
    full = backend.req_to_token[requests.long(), positions.long().clamp_min(0)]
    swa = backend.token_to_kv_pool.translate_loc_from_full_to_swa(full)
    return swa // backend.token_to_kv_pool.swa_page_size


def forward_prefill_with_checkpoints(
    backend, state_pool, x, sin, cos, compressor_op, arguments
):
    fm = backend.forward_metadata
    outputs = []
    for chunk in fm.dsv4_c4_checkpoint_chunks:
        requests = fm.dsv4_cycle_state_block_table[chunk.request : chunk.request + 1]
        position = torch.full_like(requests, chunk.position)
        if chunk.first:
            # A cached prefix is page aligned. Cold requests read the cleared
            # sentinel, never the previous request owner's mutable state.
            if chunk.position == 0:
                pages = torch.zeros_like(requests)
            else:
                if chunk.position % state_pool.swa_page_size:
                    raise ValueError(
                        "A5 C4 cached/chunked prefill must start at a page boundary"
                    )
                pages = _swa_pages(backend, requests, position - 1)
            state_pool.restore_page_checkpoint(requests, pages)
        count = chunk.end - chunk.begin
        start = chunk.output_begin
        stop = start + chunk.output_count
        # The operator may pad its compressed output. Remove that padding
        # before concatenating the real outputs in request order.
        chunk_sin = torch.cat((sin[start:stop], torch.zeros_like(sin[:1])), dim=0)
        chunk_cos = torch.cat((cos[start:stop], torch.zeros_like(cos[:1])), dim=0)
        result = compressor_op(
            x[chunk.begin : chunk.end],
            **arguments,
            rope_sin=chunk_sin,
            rope_cos=chunk_cos,
            state_block_table=requests,
            cu_seqlens=torch.tensor([0, count], dtype=torch.int32, device=x.device),
            seqused=torch.full_like(requests, count),
            start_pos=position,
        )
        outputs.append(result[: chunk.output_count])
        end = chunk.position + count
        if end % state_pool.swa_page_size == 0:
            pages = _swa_pages(backend, requests, torch.full_like(requests, end - 1))
            state_pool.save_page_checkpoint(requests, pages)
    return torch.cat(outputs, dim=0)


def checkpoint_completed_pages(backend, state_pool):
    """Snapshot completed pages in fixed-shape decode/verify graphs."""
    fm = backend.forward_metadata
    requests = fm.dsv4_cycle_state_block_table
    start = fm.start_pos
    end = start + fm.seqused
    page_size = state_pool.swa_page_size
    boundary = end // page_size * page_size
    # A verify may include a speculative suffix. Retain the four history rows
    # that belong to the checkpoint's prefix, not a later ring wraparound.
    valid = (
        (boundary > start)
        & (fm.seqused > 0)
        & (end - boundary <= state_pool.ring_size - 4)
    )
    pages = _swa_pages(backend, requests, boundary - 1)
    valid = valid & (pages > 0)
    state_pool.save_page_checkpoint(requests, pages, valid)
