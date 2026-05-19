struct Uniforms {
  view:     mat4x4<f32>,
  proj:     mat4x4<f32>,
  campos:   vec3<f32>,
  viewport: vec2<f32>,
}

// layout per Gaussian (64 floats = 256 bytes):
//   xyz      (0-2)   world position
//   f_dc     (3-5)   SH degree-0, raw
//   f_rest_R (6-20)  SH degrees 1-3 for R
//   f_rest_G (21-35) SH degrees 1-3 for G
//   f_rest_B (36-50) SH degrees 1-3 for B
//   opacity  (51)
//   cov      (52-57) xx xy xz yy yz zz
//   pad      (58-63)
@group(0) @binding(0) var<uniform>       u:       Uniforms;
@group(0) @binding(1) var<storage, read> splats:  array<array<f32, 64>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

const CORNERS = array<vec2<f32>, 4>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0,  1.0),
);

const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;

fn eval_sh(idx: u32, dir: vec3<f32>) -> vec3<f32> {
  let x = dir.x; let y = dir.y; let z = dir.z;
  let xx = x*x; let yy = y*y; let zz = z*z;
  let xy = x*y; let yz = y*z; let xz = x*z;

  // degree 0
  var col = vec3<f32>(
    SH_C0 * splats[idx][3],
    SH_C0 * splats[idx][4],
    SH_C0 * splats[idx][5],
  );

  // degree 1: R[6..8]  G[21..23]  B[36..38]
  let d1_0 = -SH_C1 * y;
  let d1_1 =  SH_C1 * z;
  let d1_2 = -SH_C1 * x;
  col += vec3<f32>(
    d1_0*splats[idx][6]  + d1_1*splats[idx][7]  + d1_2*splats[idx][8],
    d1_0*splats[idx][21] + d1_1*splats[idx][22] + d1_2*splats[idx][23],
    d1_0*splats[idx][36] + d1_1*splats[idx][37] + d1_2*splats[idx][38],
  );

  // degree 2: R[9..13]  G[24..28]  B[39..43]
  let d2_0 =  1.0925484305920792 * xy;
  let d2_1 = -1.0925484305920792 * yz;
  let d2_2 =  0.31539156525252005 * (2.0*zz - xx - yy);
  let d2_3 = -1.0925484305920792 * xz;
  let d2_4 =  0.5462742152960396 * (xx - yy);
  col += vec3<f32>(
    d2_0*splats[idx][9]  + d2_1*splats[idx][10] + d2_2*splats[idx][11] + d2_3*splats[idx][12] + d2_4*splats[idx][13],
    d2_0*splats[idx][24] + d2_1*splats[idx][25] + d2_2*splats[idx][26] + d2_3*splats[idx][27] + d2_4*splats[idx][28],
    d2_0*splats[idx][39] + d2_1*splats[idx][40] + d2_2*splats[idx][41] + d2_3*splats[idx][42] + d2_4*splats[idx][43],
  );

  // degree 3: R[14..20]  G[29..35]  B[44..50]
  let d3_0 = -0.5900435899266435 * y * (3.0*xx - yy);
  let d3_1 =  2.890611442640554  * xy * z;
  let d3_2 = -0.4570457994644658 * y * (4.0*zz - xx - yy);
  let d3_3 =  0.3731763325901154 * z * (2.0*zz - 3.0*(xx + yy));
  let d3_4 = -0.4570457994644658 * x * (4.0*zz - xx - yy);
  let d3_5 =  1.445305721320277  * z * (xx - yy);
  let d3_6 = -0.5900435899266435 * x * (xx - 3.0*yy);
  col += vec3<f32>(
    d3_0*splats[idx][14] + d3_1*splats[idx][15] + d3_2*splats[idx][16] + d3_3*splats[idx][17] + d3_4*splats[idx][18] + d3_5*splats[idx][19] + d3_6*splats[idx][20],
    d3_0*splats[idx][29] + d3_1*splats[idx][30] + d3_2*splats[idx][31] + d3_3*splats[idx][32] + d3_4*splats[idx][33] + d3_5*splats[idx][34] + d3_6*splats[idx][35],
    d3_0*splats[idx][44] + d3_1*splats[idx][45] + d3_2*splats[idx][46] + d3_3*splats[idx][47] + d3_4*splats[idx][48] + d3_5*splats[idx][49] + d3_6*splats[idx][50],
  );

  // SH values are zero-centered around 0.5: a Gaussian with only DC term and
  // f_dc=0 should render as mid-gray, not black. Without this the representable
  // color range would be asymmetric and the optimizer couldn't reach bright colors.
  return clamp(col + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
}

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
  let xyz = vec3<f32>(splats[idx][0], splats[idx][1], splats[idx][2]);

  // transform center to camera space; camera looks toward -z so visible points have z < 0
  let p_cam = (u.view * vec4<f32>(xyz, 1.0)).xyz;

  // cull points behind camera or with negligible opacity
  if p_cam.z >= 0.0 || splats[idx][51] < 1.0 / 255.0 {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    return out;
  }

  // reconstruct symmetric 3x3 world-space covariance from stored upper triangle
  let sig3d = mat3x3<f32>(
    vec3<f32>(splats[idx][52], splats[idx][53], splats[idx][54]),
    vec3<f32>(splats[idx][53], splats[idx][55], splats[idx][56]),
    vec3<f32>(splats[idx][54], splats[idx][56], splats[idx][57]),
  );

  // rotate covariance into camera space: Σ_cam = R Σ_world R^T
  // upper-left 3x3 of view matrix is the rotation R
  let R       = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  let sig_cam = R * sig3d * transpose(R);

  // first-order Taylor (Jacobian) approximation of perspective projection,
  // mapping 3D camera-space displacements to 2D pixel displacements:
  //   J = [[fx/z,  0,    -fx·x/z²],
  //        [0,     fy/z, -fy·y/z²]]
  // stored as mat3x2 (3 cols, 2 rows) — WGSL column-major
  let fx = u.proj[0][0] * u.viewport.x * 0.5;
  let fy = u.proj[1][1] * u.viewport.y * 0.5;
  let z  = p_cam.z;
  let J  = mat3x2<f32>(
    vec2<f32>(fx / z,                   0.0                    ),
    vec2<f32>(0.0,                      fy / z                 ),
    vec2<f32>(-fx * p_cam.x / (z * z), -fy * p_cam.y / (z * z)),
  );

  // project to 2D screen-space covariance (pixel²): Σ_2d = J Σ_cam J^T
  var cov2d = J * sig_cam * transpose(J);
  // low-pass regularizer (in pixel²): prevents sub-pixel Gaussians from
  // becoming numerical delta functions and keeps det > 0 for the inverse
  cov2d[0][0] += 0.3;
  cov2d[1][1] += 0.3;

  let a = cov2d[0][0];
  let b = cov2d[0][1];  // cov2d is symmetric so [0][1] == [1][0]
  let d = cov2d[1][1];

  // largest eigenvalue of symmetric 2x2: λ = (tr ± sqrt(tr²/4 - det)) / 2
  // quad must circumscribe the 3σ ellipse, so radius = 3·sqrt(λ_max)
  let mid    = (a + d) * 0.5;
  let disc   = sqrt(max((a - d) * (a - d) * 0.25 + b * b, 0.0));
  let radius = min(ceil(3.0 * sqrt(mid + disc)), 1024.0);

  // inverse of symmetric 2x2: (1/det) · [[d, -b], [-b, a]]
  // packed as vec3(inv_a, inv_b, inv_d) to pass to fragment shader
  let det = a * d - b * b;
  let inv = vec3<f32>(d / det, -b / det, a / det);

  // view direction from Gaussian center toward camera (world space),
  // matching the convention used during 3DGS training
  let dir  = normalize(xyz - u.campos);
  let clip = u.proj * vec4<f32>(p_cam, 1.0);
  let ndc  = clip.xy / clip.w;

  // offset the quad corner in NDC; uv carries the pixel offset for the fragment
  let corner = CORNERS[vi];
  let offset = corner * radius * vec2<f32>(2.0 / u.viewport.x, 2.0 / u.viewport.y);

  out.pos       = vec4<f32>(ndc + offset, clip.z / clip.w, 1.0);
  out.uv        = corner * radius;  // pixel offset from center, interpolated across quad
  out.cov2d_inv = inv;
  out.color     = eval_sh(idx, dir);
  out.opacity   = splats[idx][51];
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
