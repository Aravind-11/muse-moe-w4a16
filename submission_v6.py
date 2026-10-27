# okc: problem=muse_moe_w4a16
"""NVFP4 routed experts on H200.

S2: GEMM1 and SiLU*up are one fused swap-AB kernel. Weights sit on the MMA M side
(mma.m16n8k16 BF16), tokens on N. One warp owns 16 gate rows plus the matching 16 up
rows, so silu(gate)*up is finished in registers and written straight into the GEMM2
input. Compared with the first swap-AB try (4-8x slower than Marlin):

  * work list comes from moe_align_block_size (no per-CTA scan of the routing ids),
  * weights are repacked so each lane reads one 16-byte fragment per 64-K tile,
  * E2M1 -> bf16 is rotate+mask on the packed word, scales fold in with one multiply,
  * 5-6 stage cp.async ring, activations in padded smem (no bank conflicts),
  * fp32 SiLU epilogue, no cache1 round trip, no silu_and_mul launch.

GEMM2 and the post-norm stay on Marlin / Triton. prepare() repacks the 32 local
experts (same byte count) and compiles with load_inline (device code has no PyTorch
headers). A scored shape uses this path only when it matches Marlin under the
checks.py rule AND its CUDA-graph replay time beats Marlin's by a margin; any compile
or launch failure leaves that shape on Marlin. The align histogram covers this GPU's
32 local experts; atomic split-K stays off (measured 0.982x).
"""

from typing import Callable, TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8

# checks.py: relative L2 <= 1.5%, and every row <= 5x that, with a floor at
# 10% of the mean row norm. Inlined so prepare() does not import the reference.
REL_L2_TOL = 1.5e-2
ROW_FACTOR = 5.0
ROW_NORM_FLOOR = 0.1

# Judge shapes, in leaderboard order. Names are the load plus the token count.
SCORED_SHAPES = (
    ("decode-bs1", 1),
    ("decode-bs16", 16),
    ("decode-bs32", 32),
    ("verify-bs8", 8),
    ("verify-bs128", 128),
    ("verify-bs256", 256),
)
GATE_WARMUP = 3
GATE_ITERS = 7
GRAPH_ITERS = 21
GEMM1_EXPERT_BYTES = 8192 * (2048 + 256)   # packed weights + scales of one local expert's W13


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


# ---PACK-BEGIN---
# Bit layout of one packed 32-bit word (8 E2M1 codes, code bits = [sign e1 e0 m]).
# The kernel turns a word into four bf16x2 A registers with rotate+mask ops.
# A code c (3 magnitude bits) placed at bf16 bits c<<6 is the E2M1 value times 2**-126
# (denormal for c==1), so the scale multiply restores the exponent.
_P = (6, 7, 8, 15, 22, 23, 24, 31)      # bf16x2 bit positions: lo (m,e0,e1,sign), hi (m,e0,e1,sign)
_ROT = (0, 3, 6)                        # regs 0..2 are rotl(w, _ROT[j]) & 0x81C081C0
_SRC_REG3 = (10, 11, 13, 14, 26, 27, 29, 30)  # reg 3 uses the 8 leftover bits


def _word_sources():
    """Return, for each word bit q, the (slot, code_bit) stored there. slot = 2*reg + (0 lo, 1 hi)."""
    src = {}
    for j in range(3):
        for i in range(8):
            q = (_P[i] - _ROT[j]) % 32
            assert q not in src
            src[q] = (2 * j + (i >> 2), i & 3)
    for i in range(8):
        assert _SRC_REG3[i] not in src
        src[_SRC_REG3[i]] = (6 + (i >> 2), i & 3)
    assert sorted(src) == list(range(32))
    return [src[q] for q in range(32)]


def _pack_indices(nt_n, kt_n):
    """numpy gather indices. Packed piece order: [nt][kt][mat][lane][atom][4 bytes] then 64 scale bytes."""
    import numpy as np

    shape = (nt_n, kt_n, 2, 32, 4)
    nt = np.arange(nt_n).reshape(-1, 1, 1, 1, 1)
    kt = np.arange(kt_n).reshape(1, -1, 1, 1, 1)
    mat = np.arange(2).reshape(1, 1, -1, 1, 1)
    lane = np.arange(32).reshape(1, 1, 1, -1, 1)
    atom = np.arange(4).reshape(1, 1, 1, 1, -1)
    grp = lane >> 2
    tig = lane & 3
    half_rows = nt_n * 16
    ix = {"n": [], "kb": [], "sh": []}
    for i in range(8):
        dk = i & 1
        rsel = (i >> 1) & 1
        kh = i >> 2
        n = mat * half_rows + nt * 16 + grp + 8 * rsel
        k = kt * 64 + atom * 16 + tig * 2 + dk + 8 * kh
        n = np.broadcast_to(n, shape).astype(np.int64)
        k = np.broadcast_to(k, shape).astype(np.int64)
        ix["n"].append(n)
        ix["kb"].append(k >> 1)
        ix["sh"].append(((k & 1) * 4).astype(np.uint8))
    shape_s = (nt_n, kt_n, 2, 8, 2, 4)
    nt = np.arange(nt_n).reshape(-1, 1, 1, 1, 1, 1)
    kt = np.arange(kt_n).reshape(1, -1, 1, 1, 1, 1)
    mat = np.arange(2).reshape(1, 1, -1, 1, 1, 1)
    grp = np.arange(8).reshape(1, 1, 1, -1, 1, 1)
    half = np.arange(2).reshape(1, 1, 1, 1, -1, 1)
    atom = np.arange(4).reshape(1, 1, 1, 1, 1, -1)
    ix["sn"] = np.broadcast_to(mat * half_rows + nt * 16 + grp + 8 * half, shape_s).astype(np.int64)
    ix["sk"] = np.broadcast_to(kt * 4 + atom, shape_s).astype(np.int64)
    return ix


def _scale_code_lut():
    """e4m3 byte -> 8-bit code (e'<<3 | mant), e' = true exponent + 10, exact for every finite e4m3.

    The kernel turns a code into bf16 bits with one multiply-add: (code << 4) + (177 << 7) in both
    halves, i.e. scale * 2^60. Zero maps to 0; its blocks get their codes zeroed at repack."""
    import numpy as np

    lut = np.zeros(256, dtype=np.uint8)
    for b in range(256):
        v = b & 0x7F
        e, m = v >> 3, v & 7
        if v == 0:
            lut[b] = 0
            continue
        if e == 0:
            p = m.bit_length() - 1
            et = p - 9
            mant = int((m / (1 << p) - 1.0) * 8)
        else:
            et = e - 7
            mant = m
        lut[b] = ((et + 10) << 3) | mant
    return lut


def _pack_expert(w_e, sfc_e, ix, stack, cat):
    """w_e uint8 [2*half, K/2] (low nibble = even k), sfc_e coded scales [2*half, K/16] -> flat pieces."""
    g = [(w_e[ix["n"][i], ix["kb"][i]] >> ix["sh"][i]) & 15 for i in range(8)]
    # A zero block scale makes the block exactly zero in the reference; zero its codes too.
    g = [g[i] * (sfc_e[ix["n"][i], ix["kb"][i] >> 3] != 0) for i in range(8)]
    byts = [None] * 4
    for q, (i, cb) in enumerate(_word_sources()):
        bit = ((g[i] >> cb) & 1) << (q % 8)
        b = q // 8
        byts[b] = bit if byts[b] is None else (byts[b] | bit)
    words = stack(byts)
    nt_n, kt_n = words.shape[0], words.shape[1]
    words = words.reshape(nt_n, kt_n, 2, 512)
    sc = sfc_e[ix["sn"], ix["sk"]].reshape(nt_n, kt_n, 2, 64)
    return cat([words, sc]).reshape(-1)
# ---PACK-END---



def _repack_w13(w, sf, gs):
    """Repack 32 local experts' W13 into 16-byte-per-lane fragments. Same byte count as the input."""
    import numpy as np

    w = w[:32].contiguous()
    sf = sf[:32].contiguous()
    gs = gs[:32].contiguous().float()
    if w.shape != (32, 8192, 2048) or sf.shape != (32, 8192, 256) or gs.shape != (32,):
        raise RuntimeError(f"unexpected w13 shape {tuple(w.shape)} sf {tuple(sf.shape)} gs {tuple(gs.shape)}")
    sf_u8 = sf.view(torch.uint8)
    ix_np = _pack_indices(256, 64)
    dev = w.device
    ix = {
        "n": [torch.from_numpy(a).to(dev) for a in ix_np["n"]],
        "kb": [torch.from_numpy(a).to(dev) for a in ix_np["kb"]],
        "sh": [torch.from_numpy(a).to(dev) for a in ix_np["sh"]],
        "sn": torch.from_numpy(ix_np["sn"]).to(dev),
        "sk": torch.from_numpy(ix_np["sk"]).to(dev),
    }
    lut = torch.from_numpy(_scale_code_lut()).to(dev)
    out = torch.empty((32, 256 * 64 * 1152), dtype=torch.uint8, device=dev)
    for e in range(32):
        out[e] = _pack_expert(
            w[e], lut[sf_u8[e].long()], ix,
            lambda lst: torch.stack(lst, dim=-1),
            lambda lst: torch.cat(lst, dim=-1),
        )
    return out, gs


# Device code only. PyTorch headers stay in _CPP_SRC, which g++ compiles.
# nvcc on the judge image fails inside torch/csrc/autograd/edge.h.
_CUDA_SRC = r'''
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int KT = 64;        // K = 4096 in tiles of 64
constexpr int PIECE = 1152;   // 16 gate rows + 16 up rows, one K tile: 2 * (512 weights + 64 scales)
constexpr int ACT_LD = 72;    // bf16 per smem activation row (64 + 8 pad: conflict-free fragment reads)
constexpr int KDIM = 4096;
constexpr int INTER = 4096;
constexpr int NT16 = 256;     // 16-row units in the gate half

__device__ __forceinline__ void cp16(void* smem, const void* gmem) {
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ void mma16816(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Packed word -> four bf16x2 registers holding E2M1 magnitude/sign at bf16 bits (c << 6).
// The value is the true E2M1 number times 2^-126; the scale multiply restores it.
__device__ __forceinline__ void deq(uint32_t w, uint32_t r[4]) {
  const uint32_t M = 0x81C081C0u;
  r[0] = w & M;
  r[1] = __funnelshift_l(w, w, 3) & M;
  r[2] = __funnelshift_l(w, w, 6) & M;
  r[3] = ((w >> 4) & 0x00C000C0u) | ((w >> 5) & 0x01000100u) | ((w << 1) & 0x80008000u);
}

// Coded scale byte -> bf16x2, same value in both halves, equal to scale * 2^60.
// A = v(2^-126) * scale * 2^60 = w * 2^-66; the epilogue multiplies by gs * 2^66.
__device__ __forceinline__ uint32_t scale_dup(uint32_t u, int s) {
  uint32_t b = (u >> (8 * s)) & 0xFFu;
  return b * 0x00100010u + 0x58805880u;
}

template <int NW, int G, int S>
__global__ void __launch_bounds__(NW * 32)
gemm1_silu_kernel(const __nv_bfloat16* __restrict__ X, const uint8_t* __restrict__ W,
                  const float* __restrict__ GS, const int* __restrict__ sorted_ids,
                  const int* __restrict__ block_experts, const int* __restrict__ n_padded,
                  __nv_bfloat16* __restrict__ OUT, int numel, int snumel, int mode) {
  constexpr int T = G * 8;
  constexpr int NTH = NW * 32;
  constexpr int W_STAGE = NW * PIECE;
  constexpr int A_STAGE = T * ACT_LD * 2;
  constexpr int STAGE = W_STAGE + A_STAGE;

  const int blk = (int)blockIdx.y;
  if (blk * T >= *n_padded) return;
  const int expert = block_experts[blk];
  if (expert < 0) return;

  extern __shared__ __align__(128) unsigned char dyn[];
  char* stages = reinterpret_cast<char*>(dyn);
  int* tokoff = reinterpret_cast<int*>(dyn + S * STAGE);
  int* slots = tokoff + T;
  __shared__ int nact_s;

  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int grp = lane >> 2;
  const int tig = lane & 3;

  if (tid == 0) nact_s = 0;
  __syncthreads();
  if (tid < T) {
    const int ii = blk * T + tid;
    int slot = ii < snumel ? sorted_ids[ii] : numel;
    bool valid = slot < numel;
    slots[tid] = valid ? slot : -1;
    tokoff[tid] = valid ? (slot >> 3) * KDIM : -1;
    if (valid) atomicMax(&nact_s, tid + 1);
  }
  __syncthreads();
  const int nact = nact_s;
  if (nact <= 0) return;

  const int nchunk = (int)blockIdx.x;
  const int nt = nchunk * NW + warp;
  const uint8_t* wsrc = W + (((size_t)expert * NT16 + nt) * KT) * PIECE;

  auto issue = [&](int kt, int st) {
    char* sbase = stages + st * STAGE;
    const uint8_t* src = wsrc + (size_t)kt * PIECE;
    char* dst = sbase + warp * PIECE;
    for (int i = lane; i < PIECE / 16; i += 32) cp16(dst + i * 16, src + i * 16);
    char* adst = sbase + W_STAGE;
    for (int i = tid; i < T * 8; i += NTH) {
      int tok = i >> 3;
      int c = i & 7;
      int off = tokoff[tok];
      if (off >= 0) cp16(adst + tok * (ACT_LD * 2) + c * 16, X + off + kt * 64 + c * 8);
    }
  };

  float accg[G][4];
  float accu[G][4];
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      accg[g][i] = 0.0f;
      accu[g][i] = 0.0f;
    }
  }

#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    issue(s, s);
    cp_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<S - 2>();
    __syncthreads();
    {
      int nk = kt + S - 1;
      if (nk < KT) issue(nk, nk % S);
      cp_commit();
    }
    if (mode == 1) continue;

    const char* sbase = stages + (kt % S) * STAGE;
    const char* pg = sbase + warp * PIECE;
    const char* pu = pg + 576;
    const uint4 wg4 = *reinterpret_cast<const uint4*>(pg + lane * 16);
    const uint4 wu4 = *reinterpret_cast<const uint4*>(pu + lane * 16);
    const uint32_t wgv[4] = {wg4.x, wg4.y, wg4.z, wg4.w};
    const uint32_t wuv[4] = {wu4.x, wu4.y, wu4.z, wu4.w};
    uint32_t sc[2][2][4];  // [mat][row half][atom]
    {
      const uint32_t g0 = *reinterpret_cast<const uint32_t*>(pg + 512 + (grp * 2) * 4);
      const uint32_t g1 = *reinterpret_cast<const uint32_t*>(pg + 512 + (grp * 2 + 1) * 4);
      const uint32_t u0 = *reinterpret_cast<const uint32_t*>(pu + 512 + (grp * 2) * 4);
      const uint32_t u1 = *reinterpret_cast<const uint32_t*>(pu + 512 + (grp * 2 + 1) * 4);
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        sc[0][0][s] = scale_dup(g0, s);
        sc[0][1][s] = scale_dup(g1, s);
        sc[1][0][s] = scale_dup(u0, s);
        sc[1][1][s] = scale_dup(u1, s);
      }
    }
    const __nv_bfloat16* ab = reinterpret_cast<const __nv_bfloat16*>(sbase + W_STAGE);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint32_t rg[4], ru[4];
      deq(wgv[s], rg);
      deq(wuv[s], ru);
      const uint32_t ag0 = hmul2(rg[0], sc[0][0][s]);
      const uint32_t ag1 = hmul2(rg[1], sc[0][1][s]);
      const uint32_t ag2 = hmul2(rg[2], sc[0][0][s]);
      const uint32_t ag3 = hmul2(rg[3], sc[0][1][s]);
      const uint32_t au0 = hmul2(ru[0], sc[1][0][s]);
      const uint32_t au1 = hmul2(ru[1], sc[1][1][s]);
      const uint32_t au2 = hmul2(ru[2], sc[1][0][s]);
      const uint32_t au3 = hmul2(ru[3], sc[1][1][s]);
#pragma unroll
      for (int g = 0; g < G; ++g) {
        if (g * 8 < nact) {
          const __nv_bfloat16* bp = ab + (g * 8 + grp) * ACT_LD + s * 16 + tig * 2;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(bp);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(bp + 8);
          mma16816(accg[g], ag0, ag1, ag2, ag3, b0, b1);
          mma16816(accu[g], au0, au1, au2, au3, b0, b1);
        }
      }
    }
  }

  if (mode == 1) return;
  const float gf = GS[expert] * 73786976294838206464.0f;
  const int row0 = nchunk * (NW * 16) + warp * 16 + grp;
#pragma unroll
  for (int g = 0; g < G; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int idx = g * 8 + tig * 2 + c;
      const int slot = slots[idx];
      if (slot >= 0) {
        float x0 = accg[g][c] * gf;
        float y0 = accu[g][c] * gf;
        float x1 = accg[g][2 + c] * gf;
        float y1 = accu[g][2 + c] * gf;
        float v0 = x0 * __frcp_rn(1.0f + __expf(-x0)) * y0;
        float v1 = x1 * __frcp_rn(1.0f + __expf(-x1)) * y1;
        __nv_bfloat16* o = OUT + (size_t)slot * INTER;
        o[row0] = __float2bfloat16(v0);
        o[row0 + 8] = __float2bfloat16(v1);
      }
    }
  }
}

template <int NW, int G, int S>
struct Cfg {
  static constexpr int T = G * 8;
  static constexpr size_t smem = (size_t)S * (NW * PIECE + T * ACT_LD * 2) + (size_t)T * 8 + 128;
};

template <int NW, int G, int S>
static int run(const void* x, const void* w, const void* gs, const void* sorted_ids, const void* bexp,
               const void* npad, void* out, int numel, int snumel, int nblocks, int mode, cudaStream_t stream) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(gemm1_silu_kernel<NW, G, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)Cfg<NW, G, S>::smem);
    init = true;
  }
  dim3 grid(NT16 / NW, nblocks);
  dim3 block(NW * 32);
  gemm1_silu_kernel<NW, G, S><<<grid, block, Cfg<NW, G, S>::smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const uint8_t*>(w),
      reinterpret_cast<const float*>(gs), reinterpret_cast<const int*>(sorted_ids),
      reinterpret_cast<const int*>(bexp), reinterpret_cast<const int*>(npad),
      reinterpret_cast<__nv_bfloat16*>(out), numel, snumel, mode);
  return (int)cudaGetLastError();
}

// (bsm, variant, warps per CTA, token groups of 8, cp.async stages)
#define MUSE_VARIANTS(X) \
  X(8, 0, 4, 1, 6)  X(8, 1, 2, 1, 8)  X(8, 2, 8, 1, 4)  \
  X(16, 0, 4, 2, 6) X(16, 1, 2, 2, 8) X(16, 2, 8, 2, 4) \
  X(32, 0, 8, 4, 5) X(32, 1, 4, 4, 6) \
  X(48, 0, 8, 6, 4) X(48, 1, 4, 6, 5) \
  X(64, 0, 8, 8, 4) X(64, 1, 4, 8, 5)

extern "C" int gemm1_launch(const void* x, const void* w, const void* gs, const void* sorted_ids,
                            const void* bexp, const void* npad, void* out, int numel, int snumel, int bsm,
                            int variant, int nblocks, int mode, cudaStream_t stream) {
#define MUSE_X(B, V, NW, G, S) \
  if (bsm == B && variant == V) return run<NW, G, S>(x, w, gs, sorted_ids, bexp, npad, out, numel, snumel, nblocks, mode, stream);
  MUSE_VARIANTS(MUSE_X)
#undef MUSE_X
  return -1;
}

template <int NW, int G, int S>
static void attrs_one(int* o) {
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, gemm1_silu_kernel<NW, G, S>);
  cudaFuncSetAttribute(gemm1_silu_kernel<NW, G, S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)Cfg<NW, G, S>::smem);
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gemm1_silu_kernel<NW, G, S>, NW * 32,
                                                Cfg<NW, G, S>::smem);
  o[0] = a.numRegs;
  o[1] = (int)a.localSizeBytes;
  o[2] = NW;
  o[3] = (int)Cfg<NW, G, S>::smem;
  o[4] = blocks;
  o[5] = S;
}

extern "C" void gemm1_attrs(int bsm, int variant, int* o) {
#define MUSE_X(B, V, NW, G, S) \
  if (bsm == B && variant == V) { attrs_one<NW, G, S>(o); return; }
  MUSE_VARIANTS(MUSE_X)
#undef MUSE_X
  o[0] = -1;
}
'''

_CPP_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

extern "C" int gemm1_launch(const void* x, const void* w, const void* gs, const void* sorted_ids,
                            const void* bexp, const void* npad, void* out, int numel, int snumel, int bsm,
                            int variant, int nblocks, int mode, void* stream);
extern "C" void gemm1_attrs(int bsm, int variant, int* o);

void gemm1(torch::Tensor x, torch::Tensor w, torch::Tensor gs, torch::Tensor sorted_ids,
           torch::Tensor bexp, torch::Tensor npad, torch::Tensor out, int64_t bsm, int64_t variant, int64_t mode) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x");
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == torch::kUInt8, "w");
  TORCH_CHECK(gs.is_cuda() && gs.is_contiguous() && gs.scalar_type() == torch::kFloat32, "gs");
  TORCH_CHECK(sorted_ids.is_cuda() && sorted_ids.is_contiguous() && sorted_ids.scalar_type() == torch::kInt32, "sorted_ids");
  TORCH_CHECK(bexp.is_cuda() && bexp.is_contiguous() && bexp.scalar_type() == torch::kInt32, "block_experts");
  TORCH_CHECK(npad.is_cuda() && npad.scalar_type() == torch::kInt32 && npad.numel() >= 1, "n_padded");
  TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == torch::kBFloat16, "out");
  const int64_t M = x.size(0);
  TORCH_CHECK(x.size(1) == 4096, "K");
  TORCH_CHECK(gs.numel() == 32, "experts");
  TORCH_CHECK(w.numel() == (int64_t)32 * 256 * 64 * 1152, "w bytes");
  TORCH_CHECK(out.size(0) == M * 8 && out.size(1) == 4096, "out shape");
  TORCH_CHECK(bsm == 8 || bsm == 16 || bsm == 32 || bsm == 48 || bsm == 64, "bsm");
  const int nblocks = static_cast<int>(bexp.numel());
  int rc = gemm1_launch(x.data_ptr(), w.data_ptr(), gs.data_ptr(), sorted_ids.data_ptr(), bexp.data_ptr(),
                        npad.data_ptr(), out.data_ptr(), static_cast<int>(M * 8), static_cast<int>(sorted_ids.numel()), static_cast<int>(bsm),
                        static_cast<int>(variant), nblocks, static_cast<int>(mode), at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(rc == 0, "gemm1 launch failed rc=", rc);
}

std::vector<int64_t> attrs(int64_t bsm, int64_t variant) {
  int o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  gemm1_attrs(static_cast<int>(bsm), static_cast<int>(variant), o);
  return std::vector<int64_t>(o, o + 6);
}
'''


VARIANTS = {8: 3, 16: 3, 32: 2, 48: 2, 64: 2}   # must match MUSE_VARIANTS in _CUDA_SRC


def _compile_new_kernels(w, sf, gs):
    """Return (forward, mode_box). prepare() catches failures."""
    w_packed, gs_packed = _repack_w13(w, sf, gs)
    print(f"REPACK gemm1 bytes={w_packed.nbytes + gs_packed.nbytes} packed={tuple(w_packed.shape)}", flush=True)
    from torch.utils.cpp_extension import load_inline

    mod = load_inline(
        name="muse_s2_fused_v6",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["gemm1", "attrs"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    for bsm, nv in VARIANTS.items():
        for v in range(nv):
            regs, lmem, nw, smem_d, occ, stages = mod.attrs(bsm, v)
            print(f"KERNEL_ATTR bsm={bsm} variant={v} warps={nw} stages={stages} regs={regs} "
                  f"local_bytes={lmem} smem_dyn={smem_d} ctas_per_sm={occ}", flush=True)
    mode_box = [0]

    def _forward(data, state, variant=None):
        cfg = data["config"]
        if (
            cfg["local_start"] != 0
            or cfg["local_count"] != 32
            or cfg["d_hidden"] != 4096
            or cfg["d_expert"] != 4096
            or cfg["top_k"] != TOP_K
            or cfg["n_experts"] != N_EXPERTS
        ):
            return _marlin_forward(data, state)
        v = state["variant"].get(cfg["bs"], 0) if variant is None else variant

        def _launch(x, sorted_ids, block_experts, n_padded, cache2, bsm):
            mod.gemm1(x, w_packed, gs_packed, sorted_ids, block_experts, n_padded, cache2, bsm,
                      min(v, VARIANTS[bsm] - 1), mode_box[0])

        return _marlin_forward(data, state, gemm1=_launch)

    return _forward, mode_box


def _metrics(output, ref) -> tuple[bool, float, float]:
    """Return (ok, relative L2, worst-row relative L2) using the checks.py rule."""
    if not isinstance(output, torch.Tensor):
        return False, float("inf"), float("inf")
    if output.shape != ref.shape or output.dtype != ref.dtype:
        return False, float("inf"), float("inf")
    out = output.float()
    if not torch.isfinite(out).all():
        return False, float("inf"), float("inf")
    got = ref.float()
    diff = out - got
    err = float(diff.norm() / got.norm().clamp(min=1e-12))
    rows_diff = diff.reshape(-1, got.shape[-1]).norm(dim=-1)
    rows_ref = got.reshape(-1, got.shape[-1]).norm(dim=-1)
    floor = (ROW_NORM_FLOOR * rows_ref.mean()).clamp(min=1e-12)
    row_err = float((rows_diff / torch.maximum(rows_ref, floor)).max())
    ok = err <= REL_L2_TOL and row_err <= ROW_FACTOR * REL_L2_TOL
    return ok, err, row_err


def _random_routed(bs: int, device, hidden: int, d_expert: int, seed: int) -> Inputs:
    """One routed batch like reference.make_inputs(dist="skewed", rank="busiest") on `device`.

    Not the judge's inputs. Not used on the timed path.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hidden_states = torch.randn((bs, hidden), generator=gen, dtype=torch.float32, device="cpu")
    # A few outlier activation channels, as in the skewed distribution.
    hot = torch.randperm(hidden, generator=gen)[:8]
    hidden_states[:, hot] *= 20.0
    hidden_states = hidden_states.to(device=device, dtype=torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen)
    scores = scores + torch.rand((N_EXPERTS,), generator=gen).pow(6) * 4
    vals, idx = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((idx // 32).flatten(), minlength=N_EXPERTS // 32)
    idx = (idx - 32 * pairs.argmax()) % N_EXPERTS   # busiest GPU's experts become 0-31
    weights = vals / vals.sum(dim=-1, keepdim=True)
    return {
        "hidden_states": hidden_states,
        "topk_ids": idx.to(device=device, dtype=torch.int32),
        "topk_weights": weights.to(device=device, dtype=torch.float32),
        "config": {
            "bs": bs,
            "d_hidden": hidden,
            "d_expert": d_expert,
            "n_experts": N_EXPERTS,
            "local_start": 0,
            "local_count": 32,
            "top_k": TOP_K,
            "sharding": "ep8",
        },
    }


def _median_us(fn: Callable, data, state, warmup: int = GATE_WARMUP, iters: int = GATE_ITERS) -> float:
    """Eager CUDA-event median, in microseconds. Fallback only; the judge replays a CUDA graph."""
    for _ in range(warmup):
        fn(data, state)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn(data, state)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


_FLUSH = {}


def _graph_us(fn: Callable, data, state, iters: int = GRAPH_ITERS) -> float:
    """CUDA-graph replay median in microseconds with a cold L2, like the judge. Falls back to eager."""
    try:
        dev = data["hidden_states"].device
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn(data, state)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(data, state)
        if "buf" not in _FLUSH:
            _FLUSH["buf"] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
        flush = _FLUSH["buf"]
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            flush.zero_()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e3)
        samples.sort()
        del graph
        return samples[len(samples) // 2]
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"GRAPH_TIMING_FALLBACK {type(exc).__name__}: {exc}", flush=True)
        return _median_us(fn, data, state)


_T0 = [0.0]
PREPARE_SOFT_LIMIT_S = 230.0   # TASK.md: prepare() <= 300 s; keep shapes on Marlin past this


def _elapsed() -> float:
    import time

    return time.monotonic() - _T0[0]


def _bsm_for(bs: int) -> int:
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if bs * TOP_K / 32 / block_size_m < 0.9:
            return block_size_m
    return BLOCK_SIZE_M_CANDIDATES[-1]


def _breakdown(data, state, name):
    """Graph time of the Marlin pipeline cut after each stage; differences are per-stage costs."""
    try:
        t = [_graph_us(lambda d, s, u=u: _marlin_forward(d, s, upto=u), data, state) for u in (1, 2, 3, 4, 5)]
        print(f"BREAKDOWN shape={name} ids_align={t[0]:.2f} gemm1={t[1]-t[0]:.2f} silu={t[2]-t[1]:.2f} "
              f"gemm2={t[3]-t[2]:.2f} norm={t[4]-t[3]:.2f} total={t[4]:.2f}", flush=True)
    except Exception as exc:
        print(f"BREAKDOWN_FAIL shape={name} {type(exc).__name__}: {exc}", flush=True)


def _gate_shapes(state, device) -> None:
    """Fill state['enabled'][bs] and state['variant'][bs]. Any failure leaves that shape on Marlin."""
    new_fn = state["new_fn"]
    mode_box = state["mode_box"]
    bank = state["banks"][(0, 32)]
    hidden = state["norm_w"].shape[0]
    d_expert = bank["size_n"]
    for name, bs in SCORED_SHAPES:
        state["enabled"][bs] = False
        if _elapsed() > PREPARE_SOFT_LIMIT_S:
            print(f"GATE_SKIP shape={name} prepare_elapsed_s={_elapsed():.1f}", flush=True)
            continue
        try:
            data = _random_routed(bs, device, hidden, d_expert, seed=5252 + bs)
            local = bank["expert_map"][data["topk_ids"].long()]
            live = int(torch.unique(local[local >= 0]).numel())
            ref = _marlin_forward(data, state)
            marlin_us = _graph_us(_marlin_forward, data, state)
            _breakdown(data, state, name)
        except Exception as exc:
            print(f"GATE_FAIL shape={name} marlin {type(exc).__name__}: {exc}", flush=True)
            continue
        if new_fn is None:
            print(f"GATE shape={name} bs={bs} marlin_us={marlin_us:.2f} choice=marlin kernel=none", flush=True)
            continue
        bsm = _bsm_for(bs)
        best_v, best_us = None, float("inf")
        for v in range(VARIANTS[bsm]):
            fn = lambda d, s, v=v: new_fn(d, s, variant=v)
            try:
                ok, err, row = _metrics(fn(data, state), ref)
                us = _graph_us(fn, data, state) if ok else float("nan")
                mode_box[0] = 1
                try:
                    loads = _graph_us(fn, data, state)
                finally:
                    mode_box[0] = 0
            except Exception as exc:
                mode_box[0] = 0
                print(f"VARIANT_FAIL shape={name} bsm={bsm} variant={v} {type(exc).__name__}: {exc}", flush=True)
                continue
            print(f"VARIANT shape={name} bsm={bsm} variant={v} ok={ok} err={err:.3e} row={row:.3e} "
                  f"us={us:.2f} loads_only_us={loads:.2f} marlin_us={marlin_us:.2f}", flush=True)
            if ok and us < best_us:
                best_v, best_us = v, us
        use_new = best_v is not None and best_us < marlin_us * 0.97
        state["enabled"][bs] = use_new
        state["variant"][bs] = best_v if best_v is not None else 0
        print(f"GATE shape={name} bs={bs} live_experts={live} bsm={bsm} best_variant={best_v} "
              f"new_us={best_us:.2f} marlin_us={marlin_us:.2f} speedup={marlin_us / best_us:.3f} "
              f"choice={'new' if use_new else 'marlin'}", flush=True)


def prepare(weights):
    import time

    _T0[0] = time.monotonic()
    device = weights["w13"].device
    # Clone before Marlin. Its repack may reuse the sliced storage.
    raw_w = weights["w13"][:32].contiguous().clone()
    raw_sf = weights["w13_sf"][:32].contiguous().clone()
    raw_gs = weights["w13_gs"][:32].contiguous().clone()
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    state = {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
        "enabled": {},
        "new_fn": None,
        "mode_box": [0],
        "variant": {},
    }
    try:
        state["new_fn"], state["mode_box"] = _compile_new_kernels(raw_w, raw_sf, raw_gs)
    except Exception as exc:
        state["new_fn"] = None
        msg = str(exc)
        if len(msg) > 4000:
            msg = msg[-4000:]
        print(f"COMPILE_FAIL {type(exc).__name__}: {msg}", flush=True)
    else:
        print(f"COMPILE_OK s2-fused-swapab-v6 prepare_elapsed_s={_elapsed():.1f}", flush=True)
    try:
        _gate_shapes(state, device)
    except Exception as exc:
        state["enabled"] = {}
        state["new_fn"] = None
        print(f"GATE_ABORT {type(exc).__name__}: {exc}", flush=True)
    torch.cuda.synchronize()
    print(f"PREPARE_DONE elapsed_s={_elapsed():.1f} enabled={state['enabled']} variant={state['variant']}", flush=True)
    return state


def _marlin_forward(data, state, gemm1=None, upto=5):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break

    # Ids are local (or -1). Histogram over the local expert count, not all 256.
    sorted_ids, block_experts, n_padded = moe_align_block_size(
        local_ids, block_size_m, bank["count"])
    if upto <= 1:
        return sorted_ids

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    if gemm1 is None:
        cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
        cache1 = moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
        if upto <= 2:
            return cache1
        silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    else:
        # Fused GEMM1 + silu(gate) * up, straight into cache2.
        gemm1(x, sorted_ids, block_experts, n_padded, cache2, block_size_m)
    if upto <= 3:
        return cache2
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if upto <= 4:
        return y
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    # Shape dispatch is constant for a captured graph. Shapes that failed the
    # gate, and every batch size prepare() did not time, take Marlin.
    fn = state["new_fn"]
    if fn is not None and state["enabled"].get(data["config"]["bs"], False):
        return fn(data, state)
    return _marlin_forward(data, state)
