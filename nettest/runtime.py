"""Matched export/checkpoint evaluation settings for opt-in recipes."""

import os


def runtime_command(environment, settings):
    """Use rank-zero's logical GPU mapping, never a physical CUDA ordinal."""
    env = os.environ.copy()
    devices = environment.get("train", {}).get("devices")
    device = str(settings.get("device", "auto"))
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(devices).rstrip(",")
    if device.isdigit():
        device = f"cuda:{device}"
    # Sensitive IEEE sections must be able to disable TF32 independently.
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    return [
        f"--device={device}",
        f"--matmul-precision={settings.get('matmul-precision', 'auto')}",
    ], env
