struct Uniforms {
  view:    mat4x4<f32>,
  proj:    mat4x4<f32>,
  campos:  vec3<f32>,
}

// layout: xyz(0-2) rgb(3-5) opacity(6) cov(7-12) pad(13-15)
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> splats: array<array<f32, 16>>;

struct VertOut {
  @builtin(position) pos: vec4<f32>,
  @location(0)       col: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertOut {
  let s   = splats[vi];
  let xyz = vec3<f32>(s[0], s[1], s[2]);
  let rgb = vec3<f32>(s[3], s[4], s[5]);
  let clip = u.proj * u.view * vec4<f32>(xyz, 1.0);
  var out: VertOut;
  out.pos = clip;
  out.col = rgb;
  return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.col, 1.0);
}
