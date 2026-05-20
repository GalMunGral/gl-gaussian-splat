struct Uniforms {
  view:     mat4x4<f32>,
  proj:     mat4x4<f32>,
  campos:   vec3<f32>,
  viewport: vec2<f32>,
}

// precomp layout per Gaussian (12 floats = 48 bytes), written by the prepass compute shader:
//   [0,1]   ndc xy (screen-space center)
//   [2]     depth (ndc z)
//   [3]     radius in pixels; negative means culled
//   [4,5,6] cov2d_inv packed as (inv_a, inv_b, inv_d)
//   [7,8,9] rgb color (SH evaluated in prepass)
//   [10]    opacity
//   [11]    pad
@group(0) @binding(0) var<uniform>       u:       Uniforms;
@group(0) @binding(1) var<storage, read> precomp: array<array<f32, 12>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

const CORNERS = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0,  1.0),
);

struct VertOut {
  @builtin(position) pos:       vec4<f32>,
  @location(0)       uv:        vec2<f32>,
  @location(1)       cov2d_inv: vec3<f32>,
  @location(2)       color:     vec3<f32>,
  @location(3)       opacity:   f32,
}

@vertex
fn vs_main(
  @builtin(vertex_index)   vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertOut {
  var out: VertOut;
  let idx = indices[ii];
  let g   = precomp[idx];

  let radius = g[3];
  if radius < 0.0 {
    // culled in prepass — emit degenerate position outside NDC so rasterizer discards
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return out;
  }

  // offset quad corner in NDC; uv carries the pixel offset for the fragment
  let corner = CORNERS[vi];
  let offset = corner * radius * vec2<f32>(2.0 / u.viewport.x, 2.0 / u.viewport.y);

  out.pos       = vec4<f32>(g[0] + offset.x, g[1] + offset.y, g[2], 1.0);
  out.uv        = corner * radius;
  out.cov2d_inv = vec3<f32>(g[4], g[5], g[6]);
  out.color     = vec3<f32>(g[7], g[8], g[9]);
  out.opacity   = g[10];
  return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
  // evaluate 2D Gaussian: exp(-0.5 · p^T Σ⁻¹ p)
  // Σ⁻¹ is symmetric so the quadratic form expands to:
  //   a·px² + 2b·px·py + c·py²   where (a,b,c) = (inv_xx, inv_xy, inv_yy)
  let p     = in.uv;
  let a     = in.cov2d_inv.x;
  let b     = in.cov2d_inv.y;
  let c     = in.cov2d_inv.z;
  let power = -0.5 * (a * p.x * p.x + 2.0 * b * p.x * p.y + c * p.y * p.y);
  if power > 0.0 { discard; }
  let alpha = in.opacity * exp(power);
  if alpha < 1.0 / 255.0 { discard; }
  // premultiplied alpha for Porter-Duff "over" compositing
  return vec4<f32>(in.color * alpha, alpha);
}