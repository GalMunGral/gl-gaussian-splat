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
  const device  = await adapter.requestDevice();
  const context = canvas.getContext('webgpu')!;
  const format  = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // --- load binary ---
  const res    = await fetch('/truck.bin');
  const buffer = await res.arrayBuffer();
  const N      = buffer.byteLength / 64;
  console.log(`loaded ${N} gaussians`);

  // --- GPU storage buffer ---
  const splatBuffer = device.createBuffer({
    size:  buffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(splatBuffer, 0, buffer);

  // --- uniform buffer: view(64) + proj(64) + campos(12) + pad(4) = 144 bytes ---
  const uniformBuffer = device.createBuffer({
    size:  144,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- shader + pipeline ---
  const shader = device.createShaderModule({ code: splatWGSL });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex:    { module: shader, entryPoint: 'vs_main' },
    fragment:  { module: shader, entryPoint: 'fs_main', targets: [{ format }] },
    primitive: { topology: 'point-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: splatBuffer   } },
    ],
  });

  // --- depth texture ---
  const makeDepth = () => device.createTexture({
    size:   [canvas.width, canvas.height],
    format: 'depth24plus',
    usage:  GPUTextureUsage.RENDER_ATTACHMENT,
  });
  let depthTexture = makeDepth();
  window.addEventListener('resize', () => {
    depthTexture.destroy();
    depthTexture = makeDepth();
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
    const proj   = Camera.projMatrix(Math.PI / 3, aspect, 0.1, 1000);

    device.queue.writeBuffer(uniformBuffer, 0,   view);
    device.queue.writeBuffer(uniformBuffer, 64,  proj);
    device.queue.writeBuffer(uniformBuffer, 128, camera.getPosition());

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view:       context.getCurrentTexture().createView(),
        loadOp:     'clear',
        storeOp:    'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
      depthStencilAttachment: {
        view:            depthTexture.createView(),
        depthLoadOp:     'clear',
        depthStoreOp:    'store',
        depthClearValue: 1.0,
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(N);

    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
