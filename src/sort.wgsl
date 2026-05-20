// One bitonic compare-and-swap pass.
// Gather sort: the sort key for position j is precomp[vals[j]][2] (z_cam).
// Only vals is rearranged; precomp is read-only. This means precomp can be updated
// by the prepass without resetting the sort order — each sort just re-gathers keys.

struct SortParams {
  stride:    u32,
  blockSize: u32,
}

@group(0) @binding(0) var<uniform>             sp:      SortParams;
@group(0) @binding(1) var<storage, read>       precomp: array<array<f32, 12>>;
@group(0) @binding(2) var<storage, read_write> vals:    array<u32>;

override workgroupSize: u32 = 256;

@compute @workgroup_size(workgroupSize)
fn cs_sort(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid       = gid.x;
  let stride    = sp.stride;
  let blockSize = sp.blockSize;

  // map thread index to the lower element of its compare-swap pair
  let i = ((tid / stride) * (stride * 2u)) + (tid % stride);
  let j = i + stride;
  if j >= arrayLength(&vals) { return; }

  // gather: sort key is z_cam stored at precomp[gaussian_index][2]
  let ki = precomp[vals[i]][2];
  let kj = precomp[vals[j]][2];

  // descending sort (largest ndc.z first = furthest Gaussian first = back-to-front)
  // flip from standard ascending: even blocks are descending, odd blocks ascending
  let ascending = ((i / blockSize) & 1u) == 1u;

  if (ascending && ki > kj) || (!ascending && ki < kj) {
    let vi = vals[i]; vals[i] = vals[j]; vals[j] = vi;
  }
}