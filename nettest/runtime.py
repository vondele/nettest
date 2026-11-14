"""Matched export/checkpoint evaluation settings for opt-in recipes."""

import os


def runtime_environment(environment):
    """Use rank-zero's logical GPU mapping, never a physical CUDA ordinal."""
    env = os.environ.copy()
    devices = environment.get("train", {}).get("devices")
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(devices).rstrip(",")
    # Sensitive IEEE sections must be able to disable TF32 independently.
    env.pop("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", None)
    return env


def runtime_command(environment, settings):
    device = str(settings.get("device", "auto"))
    if device.isdigit():
        device = f"cuda:{device}"
    return [
        f"--device={device}",
        f"--matmul-precision={settings.get('matmul-precision', 'auto')}",
    ], runtime_environment(environment)
