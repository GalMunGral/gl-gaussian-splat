import { Camera } from './camera';
import splatWGSL from './splat.wgsl?raw';

async function main() {
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  window.addEventListener('resize', () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  });

  // --- WebGPU init ---
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device  = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize:              adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  const context = canvas.getContext('webgpu')!;
  const format  = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // --- load binary ---
  const res      = await fetch('/truck.bin');
  const buffer   = await res.arrayBuffer();
  const gaussians = new Float32Array(buffer);
  const N        = buffer.byteLength / 256;
  console.log(`loaded ${N} gaussians`);

  // --- GPU storage buffer ---
  const splatBuffer = device.createBuffer({
    size:  buffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(splatBuffer, 0, buffer);

  // --- index buffer (sorted each frame on CPU) ---
  const indexBuffer = device.createBuffer({
    size:  N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const indices = new Uint32Array(N);
  const zValues = new Float32Array(N);

  // --- uniform buffer: view(64) + proj(64) + campos(12) + pad(4) + viewport(8) + pad(8) = 160 bytes ---
  const uniformBuffer = device.createBuffer({
    size:  160,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- shader + pipeline ---
  const shader = device.createShaderModule({ code: splatWGSL });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex:   { module: shader, entryPoint: 'vs_main' },
    fragment: { module: shader, entryPoint: 'fs_main', targets: [{
      format,
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      },
    }]},
    primitive: { topology: 'triangle-strip' },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: splatBuffer   } },
      { binding: 2, resource: { buffer: indexBuffer   } },
    ],
  });

  // --- camera ---
  const camera = new Camera(canvas);
  let lastTime = performance.now();
  let frames   = 0;

  function frame() {
    frames++;
    const now = performance.now();
    if (now - lastTime > 1000) {
      console.log(`fps: ${frames}`);
      frames   = 0;
      lastTime = now;
    }

    const aspect = canvas.width / canvas.height;
    const view   = camera.viewMatrix();

    // CPU sort: compute camera-space z for each Gaussian, sort back-to-front
    // view is column-major; row 2 of view matrix gives the z_cam of each point
    const r2 = view[2], r6 = view[6], r10 = view[10], r14 = view[14];
    for (let i = 0; i < N; i++) {
      const b   = i * 64;  // 64 floats per Gaussian
      zValues[i] = r2 * gaussians[b] + r6 * gaussians[b + 1] + r10 * gaussians[b + 2] + r14;
      indices[i] = i;
    }
    indices.sort((a, b) => zValues[a] - zValues[b]);
    device.queue.writeBuffer(indexBuffer, 0, indices);
    const proj   = Camera.projMatrix(Math.PI / 3, aspect, 0.1, 1000);

    device.queue.writeBuffer(uniformBuffer, 0,   view);
    device.queue.writeBuffer(uniformBuffer, 64,  proj);
    device.queue.writeBuffer(uniformBuffer, 128, camera.getPosition());
    device.queue.writeBuffer(uniformBuffer, 144, new Float32Array([canvas.width, canvas.height]));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       context.getCurrentTexture().createView(),
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4, N);

    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
