# app/callbacks/epoch_stats_callback.py

import os
import time
import json
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_ENABLED = True
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except (ImportError, pynvml.NVMLError):
    _GPU_ENABLED = False

class EpochStatsCallback:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._epoch_start = None
        self._gpu_readings = []
        self._ram_readings = []
        self._loss_readings = []
        self.epoch_index = 0

        parent = os.path.dirname(self.filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def on_epoch_begin(self, trainer):
        self._epoch_start = time.time()
        self._gpu_readings.clear()
        self._ram_readings.clear()
        self._loss_readings.clear()

    def on_batch_end(self, trainer, loss_value: float):
        # Leer RAM
        ram_pct = psutil.virtual_memory().percent
        # Leer GPU si está disponible
        if _GPU_ENABLED:
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
            gpu_pct = util.gpu
        else:
            gpu_pct = None

        self._loss_readings.append(loss_value)
        self._ram_readings.append(ram_pct)
        self._gpu_readings.append(gpu_pct)

    def on_epoch_end(self, trainer, epoch_loss: float):
        duration = time.time() - self._epoch_start

        # Promedios
        avg_loss = epoch_loss
        avg_ram = float(sum(self._ram_readings) / len(self._ram_readings)) if self._ram_readings else None
        if _GPU_ENABLED and self._gpu_readings:
            avg_gpu = float(sum(self._gpu_readings) / len(self._gpu_readings))
        else:
            avg_gpu = None

        record = {
            "epoch_index": self.epoch_index,
            "duration_sec": round(duration, 4),
            "avg_loss": round(avg_loss, 4),
            "avg_gpu_pct": round(avg_gpu,   4) if avg_gpu is not None else None,
            "avg_ram_pct": round(avg_ram,   4) if avg_ram is not None else None
        }

        with open(self.filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

        self.epoch_index += 1
