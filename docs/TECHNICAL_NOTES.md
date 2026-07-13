# Technical Notes

## CarRacing Rendering from the Devcontainer

The container forwards `DISPLAY` and mounts `/tmp/.X11-unix` so Gymnasium can open a host window. Before starting `CarRacing-v3` with `render_mode="human"`, allow the container's root user to access the host X server:

```bash
xhost +local:root
```

Revoke that permission when it is no longer needed:

```bash
xhost -local:root
```

CarRacing may print these messages inside the container:

```text
libGL error: glx: failed to create dri3 screen
libGL error: failed to load driver: nouveau
```

They can be ignored if the environment window renders correctly. The container uses the NVIDIA runtime and host NVIDIA driver; it does not require the open-source `nouveau` driver for CUDA execution.

## GPU Monitoring

Monitor utilization and memory from the host while an experiment runs:

```bash
watch -n1 nvidia-smi
```

A historical image-reconstruction experiment reached approximately 80% GPU utilization with `batch_size=128`. Treat this only as a previous observation: usable batch size depends on the current architecture, image size, and available VRAM.

## Legacy Experiment Commands

The previous README recorded an image-reconstruction run with batch size 64, 100 epochs, and output under `runs/imageReconstructor/test1`. The related training and parameter-inspection scripts remain under `scripts/`, but currently import the obsolete `app.*` package and must be migrated to `rwm.*` before use.
