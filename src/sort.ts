import radixSortWGSL from './radix_sort.wgsl?raw';

export const WORKGROUP_SIZE = 256;

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export class GpuSort {
  private pipelines:      { hist: GPUComputePipeline; scan: GPUComputePipeline; scatter: GPUComputePipeline };
  private wgOffsetsBuf:   GPUBuffer;
  private bktOffsetsBuf:  GPUBuffer;
  private valsScratch:    GPUBuffer;
  private passBindGroups: GPUBindGroup[];
  private numWorkgroups:  number;

  constructor(
    device:        GPUDevice,
    precompBuffer: GPUBuffer,
    valsBuffer:    GPUBuffer,  // result ends up here after 4 passes
    N:             number,
  ) {
    const numWorkgroups = Math.ceil(N / WORKGROUP_SIZE);
    this.numWorkgroups  = numWorkgroups;

    this.wgOffsetsBuf  = device.createBuffer({ size: numWorkgroups * 256 * 4, usage: GPUBufferUsage.STORAGE });
    this.bktOffsetsBuf = device.createBuffer({ size: 256 * 4,                usage: GPUBufferUsage.STORAGE });
    this.valsScratch   = device.createBuffer({ size: N * 4,                  usage: GPUBufferUsage.STORAGE });

    const module = device.createShaderModule({ code: radixSortWGSL });

    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } },
    ]});

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.pipelines = {
      hist:    device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: 'histogram' } }),
      scan:    device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: 'scan'      } }),
      scatter: device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: 'scatter'   } }),
    };

    // ping-pong: even passes read valsBuffer write valsScratch, odd passes the reverse
    const src = [valsBuffer,       this.valsScratch, valsBuffer,       this.valsScratch];
    const dst = [this.valsScratch, valsBuffer,       this.valsScratch, valsBuffer      ];

    this.passBindGroups = Array.from({ length: 4 }, (_, pass) => {
      const uniform = device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(uniform, 0, new Uint32Array([pass, N, numWorkgroups]));
      return device.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: uniform             } },
        { binding: 1, resource: { buffer: precompBuffer       } },
        { binding: 2, resource: { buffer: src[pass]           } },
        { binding: 3, resource: { buffer: this.wgOffsetsBuf  } },
        { binding: 4, resource: { buffer: this.bktOffsetsBuf } },
        { binding: 5, resource: { buffer: dst[pass]           } },
      ]});
    });
  }

  encode(encoder: GPUCommandEncoder): void {
    for (let pass = 0; pass < 4; pass++) {
      let cp = encoder.beginComputePass();
      cp.setPipeline(this.pipelines.hist);
      cp.setBindGroup(0, this.passBindGroups[pass]);
      cp.dispatchWorkgroups(this.numWorkgroups);
      cp.end();

      cp = encoder.beginComputePass();
      cp.setPipeline(this.pipelines.scan);
      cp.setBindGroup(0, this.passBindGroups[pass]);
      cp.dispatchWorkgroups(1);
      cp.end();

      cp = encoder.beginComputePass();
      cp.setPipeline(this.pipelines.scatter);
      cp.setBindGroup(0, this.passBindGroups[pass]);
      cp.dispatchWorkgroups(this.numWorkgroups);
      cp.end();
    }
  }
}
