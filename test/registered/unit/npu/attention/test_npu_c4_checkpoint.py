import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dsv4.c4_checkpoint import (
    build_checkpoint_chunks,
    checkpoint_completed_pages,
    forward_prefill_with_checkpoints,
)
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import NPUCompressStatePool
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="base-a-test-npu")


def make_pool(checkpoints=True):
    return NPUCompressStatePool(
        size=40, ring_size=8, overlap=True, head_dim=2,
        dtype=torch.float32, device="cpu", enable_memory_saver=False,
        ratio=4, request_scoped=False, swa_page_size=128,
        checkpoint_size=200 if checkpoints else None,
    )


class TestA5C4Checkpoint(unittest.TestCase):
    def test_same_page_number_does_not_overwrite_live_request(self):
        pool = make_pool()
        pool.state_cache_3d[4].fill_(7)
        pool.save_page_checkpoint(torch.tensor([4]), torch.tensor([4]))
        pool.page_checkpoints.state_cache_3d[4].fill_(11)
        self.assertTrue(torch.all(pool.state_cache_3d[4] == 7))
        pool.restore_page_checkpoint(torch.tensor([2]), torch.tensor([4]))
        self.assertTrue(torch.all(pool.state_cache_3d[2] == 11))
        self.assertTrue(torch.all(pool.state_cache_3d[4] == 7))
        self.assertNotEqual(pool.checkpoint_buffer_info().pages.data_ptr(),
                            pool.state_cache_3d.data_ptr())

    def test_cold_request_clears_previous_owner(self):
        pool = make_pool()
        pool.state_cache_3d[4].fill_(float("nan"))
        pool.restore_page_checkpoint(torch.tensor([4]), torch.tensor([0]))
        state = pool.state_cache_3d[4]
        half = state.shape[-1] // 2
        self.assertTrue(torch.all(state[..., :half] == 0))
        self.assertTrue(torch.all(torch.isneginf(state[..., half:])))

    def test_a3_layout_unchanged(self):
        pool = make_pool(checkpoints=False)
        self.assertIsNone(pool.page_checkpoints)
        self.assertEqual(pool.checkpoint_buffer_info().pages.data_ptr(),
                         pool.state_cache_3d.data_ptr())

    def test_chunk_plan_handles_ragged_and_cached_prefixes(self):
        chunks = build_checkpoint_chunks([0, 0, 259, 388], [0, 0, 2048], 128)
        self.assertEqual([(c.request, c.position, c.end-c.begin) for c in chunks],
                         [(1, 0, 128), (1, 128, 128), (1, 256, 3),
                          (2, 2048, 128), (2, 2176, 1)])
        self.assertEqual(sum(c.output_count for c in chunks), 96)
        self.assertEqual([c.output_begin for c in chunks], [0, 32, 64, 64, 96])

    @staticmethod
    def backend(requests, lengths, prefixes):
        table = torch.arange(1024).repeat(5, 1) + 128
        return SimpleNamespace(
            req_to_token=table,
            token_to_kv_pool=SimpleNamespace(
                swa_page_size=128, translate_loc_from_full_to_swa=lambda x: x),
            forward_metadata=SimpleNamespace(
                dsv4_cycle_state_block_table=torch.tensor(requests, dtype=torch.int32),
                dsv4_c4_checkpoint_chunks=build_checkpoint_chunks(lengths, prefixes, 128),
            ),
        )

    def test_large_prefill_captures_each_boundary_not_final_state(self):
        pool = make_pool()
        backend = self.backend([4], [0, 259], [0])
        tokens = torch.arange(1, 260, dtype=torch.float32).reshape(-1, 1)

        def compressor(x, *, state_block_table, start_pos, **kwargs):
            bank = int(state_block_table[0])
            start = int(start_pos[0])
            result = []
            for i, token in enumerate(x):
                pool.state_cache_3d[bank, (start+i) % 8].fill_(float(token))
                if (start+i+1) % 4 == 0:
                    result.append(token)
            # Simulate the fused operator's padding; it must not enter KV.
            return torch.stack(result + [torch.tensor([-999.0])])

        rope = torch.zeros(66, 2)
        output = forward_prefill_with_checkpoints(
            backend, pool, tokens, rope, rope, compressor, {})
        self.assertTrue(torch.equal(output[:, 0], torch.arange(4, 257, 4).float()))
        self.assertTrue(torch.equal(pool.page_checkpoints.state_cache_3d[1, :, 0],
                                    torch.arange(121, 129).float()))
        self.assertTrue(torch.equal(pool.page_checkpoints.state_cache_3d[2, :, 0],
                                    torch.arange(249, 257).float()))
        # Another request can resume the earlier prefix after the producer moved on.
        pool.restore_page_checkpoint(torch.tensor([2]), torch.tensor([1]))
        self.assertTrue(torch.equal(pool.state_cache_3d[2, :, 0],
                                    torch.arange(121, 129).float()))

    def test_decode_keeps_dummy_and_partial_pages_unchanged(self):
        pool = make_pool()
        backend = self.backend([1, 2, 0], [0, 1, 2, 2], [127, 126, 0])
        backend.forward_metadata.start_pos = torch.tensor([127, 126, 0])
        backend.forward_metadata.seqused = torch.tensor([1, 1, 0])
        pool.state_cache_3d[1].fill_(7)
        pool.state_cache_3d[2].fill_(19)
        sentinel = pool.page_checkpoints.state_cache_3d[0].clone()
        untouched = pool.page_checkpoints.state_cache_3d[2].clone()
        checkpoint_completed_pages(backend, pool)
        self.assertTrue(torch.all(pool.page_checkpoints.state_cache_3d[1] == 7))
        self.assertTrue(torch.equal(pool.page_checkpoints.state_cache_3d[0], sentinel))
        self.assertTrue(torch.equal(pool.page_checkpoints.state_cache_3d[2], untouched))


if __name__ == "__main__":
    unittest.main()
