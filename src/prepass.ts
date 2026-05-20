import prepassWGSL from './prepass.wgsl?raw';

export class Prepass {
  private pipeline:      GPUComputePipeline;
  private bindGroup:     GPUBindGroup;
  private Npadded:       number;
  private workgroupSize: number;

  constructor(
    device:        GPUDevice,
    uniformBuffer: GPUBuffer,
    splatBuffer:   GPUBuffer,
    precompBuffer: GPUBuffer,
    Npadded:       number,
    workgroupSize: number,
  ) {
    this.Npadded       = Npadded;
    this.workgroupSize = workgroupSize;

    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'          } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      layout:  device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: device.createShaderModule({ code: prepassWGSL }), entryPoint: 'cs_prepass', constants: { workgroupSize } },
    });
    this.bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: splatBuffer   } },
        { binding: 2, resource: { buffer: precompBuffer } },
      ],
    });
  }

  encode(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.Npadded / this.workgroupSize));
    pass.end();
  }
}