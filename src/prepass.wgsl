// Compute prepass: one thread per Gaussian.
// Runs only when the camera moves. Caches everything the vertex shader would otherwise
// recompute 4× per Gaussian per frame: SH color, 2D covariance projection, NDC position.
//
// Outputs:
//   precomp[i] — 12 floats of cached per-Gaussian render data (see layout below)
//               precomp[i][2] = z_cam, used as sort key by the gather sort

struct Uniforms {
  view:     mat4x4<f32>,
  proj:     mat4x4<f32>,
  campos:   vec3<f32>,
  viewport: vec2<f32>,
}

// precomp layout per Gaussian (12 floats = 48 bytes):
//   [0,1]   ndc xy (screen-space center)
//   [2]     ndc.z in [0,1]; sort key (descending = back-to-front); -1.0 sentinel for culled/padding
//   [3]     radius in pixels; negative means culled — vertex shader skips rendering
//   [4,5,6] cov2d_inv packed as (inv_a, inv_b, inv_d)
//   [7,8,9] rgb color from SH evaluation
//   [10]    opacity
//   [11]    pad
@group(0) @binding(0) var<uniform>             u:       Uniforms;
@group(0) @binding(1) var<storage, read>       splats:  array<array<f32, 64>>;
@group(0) @binding(2) var<storage, read_write> precomp: array<array<f32, 12>>;

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

  return clamp(col + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
}

override workgroupSize: u32 = 256;

@compute @workgroup_size(workgroupSize)
fn cs_prepass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i    = gid.x;
  let Npad = arrayLength(&precomp);
  let N    = arrayLength(&splats);
  if i >= Npad { return; }

  if i >= N {
    precomp[i][2] = -1.0;  // sentinel: sorts last in descending ndc.z order
    return;
  }

  let xyz   = vec3<f32>(splats[i][0], splats[i][1], splats[i][2]);
  let p_cam = (u.view * vec4<f32>(xyz, 1.0)).xyz;

  // cull: behind near plane, beyond far plane, or negligible opacity
  if p_cam.z >= -0.1 || p_cam.z < -100.0 || splats[i][51] < 1.0 / 255.0 {
    precomp[i][2] = -1.0;  // sentinel: sorts last in descending ndc.z order
    precomp[i][3] = -1.0;
    return;
  }

  // reconstruct symmetric 3x3 world-space covariance from stored upper triangle
  let sig3d = mat3x3<f32>(
    vec3<f32>(splats[i][52], splats[i][53], splats[i][54]),
    vec3<f32>(splats[i][53], splats[i][55], splats[i][56]),
    vec3<f32>(splats[i][54], splats[i][56], splats[i][57]),
  );

  // rotate covariance into camera space: Σ_cam = R Σ_world R^T
  let R       = mat3x3<f32>(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);
  let sig_cam = R * sig3d * transpose(R);

  // first-order Taylor (Jacobian) approximation of perspective projection,
  // mapping 3D camera-space displacements to 2D pixel displacements
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
  // low-pass regularizer: prevents sub-pixel Gaussians becoming numerical delta functions
  cov2d[0][0] += 0.3;
  cov2d[1][1] += 0.3;

  let a = cov2d[0][0];
  let b = cov2d[0][1];  // symmetric: [0][1] == [1][0]
  let d = cov2d[1][1];

  // largest eigenvalue → bounding radius in pixels (3σ cutoff)
  let mid    = (a + d) * 0.5;
  let disc   = sqrt(max((a - d) * (a - d) * 0.25 + b * b, 0.0));
  let radius = min(ceil(3.0 * sqrt(mid + disc)), 128.0);

  // inverse of symmetric 2x2: (1/det) · [[d,-b],[-b,a]]
  let det = a * d - b * b;
  let inv = vec3<f32>(d / det, -b / det, a / det);

  // SH color: view direction from Gaussian center toward camera (world space)
  let dir   = normalize(xyz - u.campos);
  let color = eval_sh(i, dir);

  // clip → NDC; ndc.z in [0,1] serves as both vertex depth and sort key (descending = back-to-front)
  let clip = u.proj * vec4<f32>(p_cam, 1.0);
  let ndc  = clip.xyz / clip.w;

  precomp[i][0]  = ndc.x;
  precomp[i][1]  = ndc.y;
  precomp[i][2]  = ndc.z;
  precomp[i][3]  = radius;
  precomp[i][4]  = inv.x;
  precomp[i][5]  = inv.y;
  precomp[i][6]  = inv.z;
  precomp[i][7]  = color.r;
  precomp[i][8]  = color.g;
  precomp[i][9]  = color.b;
  precomp[i][10] = splats[i][51];
}
