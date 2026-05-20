import { Camera } from './camera';
import { Prepass } from './prepass';
import { GpuSort, nextPow2, WORKGROUP_SIZE } from './sort';
import splatWGSL from './splat.wgsl?raw';

async function main() {
  const RENDER_SCALE = 0.5;  // render at half resolution; browser upscales via CSS

  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  canvas.style.width  = '100vw';
  canvas.style.height = '100vh';
  canvas.width  = Math.floor(window.innerWidth  * RENDER_SCALE);
  canvas.height = Math.floor(window.innerHeight * RENDER_SCALE);
  window.addEventListener('resize', () => {
    canvas.width  = Math.floor(window.innerWidth  * RENDER_SCALE);
    canvas.height = Math.floor(window.innerHeight * RENDER_SCALE);
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
  const base = import.meta.env.VITE_DATA_BASE ?? '.';
  const { chunks: nChunks } = await fetch(`${base}/truck.json`).then(r => r.json());

  const loadingEl = document.getElementById('loading')!;
  const barEl     = document.getElementById('loading-bar') as HTMLElement;
  const labelEl   = document.getElementById('loading-label')!;
  let done = 0;
  const parts = await Promise.all(
    Array.from({ length: nChunks }, (_, i) =>
      fetch(`${base}/truck_${i}.bin`).then(r => r.arrayBuffer()).then(buf => {
        done++;
        barEl.style.width  = `${(done / nChunks) * 100}%`;
        labelEl.textContent = `loading… ${done} / ${nChunks}`;
        return buf;
      })
    )
  );
  loadingEl.remove();

  const totalBytes = parts.reduce((s, p) => s + p.byteLength, 0);
  const combined   = new Uint8Array(totalBytes);
  let off = 0;
  for (const part of parts) { combined.set(new Uint8Array(part), off); off += part.byteLength; }
  const buffer = combined.buffer;
  const N      = totalBytes / 256;
  console.log(`loaded ${N} gaussians (${nChunks} chunks)`);

  // --- GPU buffers ---
  const Npadded = nextPow2(N);

  const splatBuffer = device.createBuffer({
    size:  buffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(splatBuffer, 0, buffer);

  // precomp: 12 floats per Gaussian (Npadded entries; padding slots get z_cam=1e9 sentinel)
  // precomp[i][2] = z_cam, used as the gather-sort key
  const precompBuffer = device.createBuffer({
    size:  Npadded * 12 * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // index buffer: initialised to identity [0,1,...,Npadded-1]; only the sort rearranges it
  const indexBuffer = device.createBuffer({
    size:  Npadded * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, new Uint32Array(Npadded).map((_, i) => i));

  // --- uniform buffer: view(64) + proj(64) + campos(12) + pad(4) + viewport(8) + pad(8) = 160 bytes ---
  const uniformBuffer = device.createBuffer({
    size:  160,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- prepass + sort ---
  const prepass = new Prepass(device, uniformBuffer, splatBuffer, precompBuffer, Npadded, WORKGROUP_SIZE);
  const sorter  = new GpuSort(device, precompBuffer, indexBuffer, N);

  // --- render pipeline ---
  const shader = device.createShaderModule({ code: splatWGSL });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform'           } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage'  } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage'  } },
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
      { binding: 0, resource: { buffer: uniformBuffer  } },
      { binding: 1, resource: { buffer: precompBuffer  } },
      { binding: 2, resource: { buffer: indexBuffer    } },
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
    const proj   = Camera.projMatrix(Math.PI / 3, aspect, 0.1, 100);

    device.queue.writeBuffer(uniformBuffer, 0,   view);
    device.queue.writeBuffer(uniformBuffer, 64,  proj);
    device.queue.writeBuffer(uniformBuffer, 128, camera.getPosition());
    device.queue.writeBuffer(uniformBuffer, 144, new Float32Array([canvas.width, canvas.height]));

    const encoder = device.createCommandEncoder();
    if (camera.needsPrepass) {
      prepass.encode(encoder);
      camera.needsPrepass = false;
    }
    if (camera.needsSort) {
      sorter.encode(encoder);
      camera.needsSort = false;
    }

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