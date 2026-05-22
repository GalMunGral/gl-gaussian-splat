import { mat4 } from 'gl-matrix';

export class Camera {
  target   = new Float32Array([0, 0, 0]);
  distance = 6;
  yaw      = 0;
  pitch    = 0.4;

  needsPrepass = true;
  needsSort    = true;

  private dragging = false;
  private lastX    = 0;
  private lastY    = 0;

  constructor(canvas: HTMLCanvasElement) {
    canvas.addEventListener('mousedown', e => {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { this.dragging = false; });
    window.addEventListener('mousemove', e => {
      if (!this.dragging) return;
      this.yaw   -= (e.clientX - this.lastX) * 0.005;
      this.pitch  += (e.clientY - this.lastY) * 0.005;
      this.pitch   = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
      this.lastX   = e.clientX;
      this.lastY   = e.clientY;
      this.needsPrepass = true;
      this.needsSort    = true;
    });
    canvas.addEventListener('wheel', e => {
      this.distance *= Math.exp(e.deltaY * 0.001);
      this.distance  = Math.max(3, Math.min(20, this.distance));
      this.needsPrepass = true;  // NDC positions and SH view dirs change with campos
    }, { passive: true });
  }

  getPosition(): Float32Array {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw),   sy = Math.sin(this.yaw);
    return new Float32Array([
      this.target[0] + this.distance * cp * sy,
      this.target[1] + this.distance * sp,
      this.target[2] + this.distance * cp * cy,
    ]);
  }

  viewMatrix(): Float32Array {
    const pos  = this.getPosition();
    const view = mat4.create();
    mat4.lookAt(view, pos, this.target, [0, 1, 0]);
    return view as Float32Array;
  }

  static projMatrix(fovY: number, aspect: number, near: number, far: number): Float32Array {
    const proj = mat4.create();
    mat4.perspectiveZO(proj, fovY, aspect, near, far);
    return proj as Float32Array;
  }
}