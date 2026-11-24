# HY4 Flash and formal iHC operators

The `hcu_adapter.hy4_flash_ops` package provides the community Triton Engram, IndexPool and iHC operators for the external HY4 Flash model. The formal HY4 text model reuses iHC. Model registration, quantization mapping, KV policy and serving hooks belong to sglang-plugin-das.

Source: sgl-project/sglang main `2544c9ac34be4447bdffa612c37015a3f8da9e84`. Baseline adapter: `94911d1d776a315ce3989820dd3b9c5b1e8cccb3`; official SGLang v0.5.22: `baa9befa381511cd4d56f83980951e5a45fffc6d`. Exact port hashes are recorded in `hy4-flash-ops-manifest.json`; model source snapshots are in the plugin docs. These operators do not replace framework source files.

On 2026-11-23 gfx938, independent numerical and changed-input Graph checks passed for iHC, fused norm, Engram BF16/FP8 lookup, causal paged/ragged windows, IndexPool logits/query/write/update. The formal model additionally passed iHC `[T,4,6144]` BF16 checks at T=0/1/4/65 on 2026-11-24. Full-model evidence: Flash Eager/Graph 9/9 and GSM8K 193/200; formal Eager/Graph 9/9 and GSM8K 196/200, under the distinct model recipes documented by the plugin. These are historical checks, not newly rerun operator benchmarks or upstream equivalence claims.

The retained deployment checks are `py_scr/test_hy4_flash_{kernels,indexpool,pool_update,fused_norm}.py`, run after selecting the adapter/plugin/official source roots. Raw evidence is outside Git under `flash_validation_20261123/` and `hy4_validation_20261124/formal_port/`. Native adapter extensions must match the local Torch/DTK ABI. Generated hipify `.hip`/`.cuh` build outputs are excluded from this change.
