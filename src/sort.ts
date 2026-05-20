import sortWGSL from './sort.wgsl?raw';

export const WORKGROUP_SIZE = 256;

interface Pass { stride: number; blockSize: number; }

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Enumerate all (stride, blockSize) passes for a bitonic sort of Npadded elements.
// For each block-size doubling step k, we need k swap passes with strides
// blockSize/2, blockSize/4, ..., 1.
function buildPasses(Npadded: number): Pass[] {
  const passes: Pass[] = [];
  for (let blockSize = 2; blockSize <= Npadded; blockSize <<= 1) {
    for (let stride = blockSize >> 1; stride >= 1; stride >>= 1) {
      passes.push({ stride, blockSize });
    }
  }
  return passes;
}

export class GpuSort {
  private sortPipeline:   GPUComputePipeline;
  private sortBindGroups: GPUBindGroup[];
  private passes:         Pass[];
  private Npadded:        number;

  constructor(
    device:        GPUDevice,
    precompBuffer: GPUBuffer,  // read-only; sort gathers z_cam from precomp[val][2]
    valsBuffer:    GPUBuffer,  // read+write; sort rearranges indices in place
    N:             number,
  ) {
    this.Npadded = nextPow2(N);
    this.passes  = buildPasses(this.Npadded);

    const sortBGL = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'          } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } },
      ],
    });
    this.sortPipeline = device.createComputePipeline({
      layout:  device.createPipelineLayout({ bindGroupLayouts: [sortBGL] }),
      compute: { module: device.createShaderModule({ code: sortWGSL }), entryPoint: 'cs_sort', constants: { workgroupSize: WORKGROUP_SIZE } },
    });

    // Pre-create one uniform buffer + bind group per pass so encode() never calls writeBuffer.
    this.sortBindGroups = this.passes.map(({ stride, blockSize }) => {
      const paramsBuf = device.createBuffer({
        size:  8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([stride, blockSize]));
      return device.createBindGroup({
        layout: sortBGL,
        entries: [
          { binding: 0, resource: { buffer: paramsBuf     } },
          { binding: 1, resource: { buffer: precompBuffer } },
          { binding: 2, resource: { buffer: valsBuffer    } },
        ],
      });
    });
  }

  // Encode all bitonic passes into an existing command encoder.
  // Must run after the prepass (which writes keys and vals) and before the render pass.
  encode(encoder: GPUCommandEncoder): void {
    const workgroups = Math.ceil(this.Npadded / 2 / WORKGROUP_SIZE);
    for (const bg of this.sortBindGroups) {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.sortPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }
  }
}