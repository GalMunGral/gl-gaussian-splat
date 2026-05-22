// LSD radix sort — one pass over one byte of the key.
// Run 4 times (bytes 0–3, LSB→MSB) to fully sort by a u32 key.
//
// Key encoding:  bitcast<u32>(precomp[val][2]) ^ 0xFFFFFFFF
//   ndc.z ∈ [0,1] are positive floats whose IEEE 754 bit patterns already
//   order correctly as u32. XOR flips all bits → descending order.
//   Sentinel ndc.z = −1.0 → XOR = 0x407FFFFF < any valid ndc.z XOR,
//   so culled Gaussians naturally sink to the end.

struct Params {
  byte:          u32,
  N:             u32,
  num_workgroups: u32,
}

@group(0) @binding(0) var<uniform>             p:           Params;
@group(0) @binding(1) var<storage, read>       precomp:     array<array<f32, 12>>;
@group(0) @binding(2) var<storage, read>       vals_in:     array<u32>;
@group(0) @binding(3) var<storage, read_write> wg_offsets:  array<u32>;  // [num_workgroups * 256]
@group(0) @binding(4) var<storage, read_write> bkt_offsets: array<u32>;  // [256]
@group(0) @binding(5) var<storage, read_write> vals_out:    array<u32>;

fn key_byte(idx: u32) -> u32 {
  return ((bitcast<u32>(precomp[idx][2]) ^ 0xFFFFFFFFu) >> (p.byte * 8u)) & 0xFFu;
}

// ── 1. Histogram ─────────────────────────────────────────────────────────────

var<workgroup> sh_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn histogram(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
) {
  atomicStore(&sh_hist[lid.x], 0u);
  workgroupBarrier();

  if gid.x < p.N {
    atomicAdd(&sh_hist[key_byte(vals_in[gid.x])], 1u);
  }
  workgroupBarrier();

  wg_offsets[wid.x * 256u + lid.x] = atomicLoad(&sh_hist[lid.x]);
}

// ── 2. Scan ───────────────────────────────────────────────────────────────────
// Thread b walks column b of wg_offsets, computing an exclusive prefix sum
// in place. Column totals land in sh_totals[b].
// Thread 0 then scans sh_totals sequentially → bkt_offsets.

var<workgroup> sh_totals: array<u32, 256>;

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid: vec3<u32>) {
  let b = lid.x;

  var running = 0u;
  for (var wg = 0u; wg < p.num_workgroups; wg++) {
    let i            = wg * 256u + b;
    let count        = wg_offsets[i];
    wg_offsets[i]    = running;
    running         += count;
  }
  sh_totals[b] = running;
  workgroupBarrier();

  if b == 0u {
    var acc = 0u;
    for (var i = 0u; i < 256u; i++) {
      let count   = sh_totals[i];
      sh_totals[i] = acc;
      acc         += count;
    }
  }
  workgroupBarrier();

  bkt_offsets[b] = sh_totals[b];
}

// ── 3. Scatter ────────────────────────────────────────────────────────────────
// Thread b scans sh_buckets in thread-index order, assigning ranks to elements
// whose bucket == b. Each thread writes to disjoint slots of sh_rank — stable,
// no atomics needed.
//
// Output position = bkt_offsets[bucket]              (elements in earlier buckets globally)
//                 + wg_offsets[wg * 256 + bucket]    (same bucket, earlier workgroups)
//                 + sh_rank[lid]                     (rank within this workgroup)

var<workgroup> sh_buckets: array<u32, 256>;
var<workgroup> sh_rank:    array<u32, 256>;

@compute @workgroup_size(256)
fn scatter(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
) {
  let live      = gid.x < p.N;
  let idx       = vals_in[gid.x];
  let bucket    = select(256u, key_byte(idx), live);
  sh_buckets[lid.x] = bucket;
  workgroupBarrier();

  let b    = lid.x;
  var rank = 0u;
  for (var t = 0u; t < 256u; t++) {
    if sh_buckets[t] == b {
      sh_rank[t] = rank;
      rank++;
    }
  }
  workgroupBarrier();

  if live {
    vals_out[bkt_offsets[bucket] + wg_offsets[wid.x * 256u + bucket] + sh_rank[lid.x]] = idx;
  }
}