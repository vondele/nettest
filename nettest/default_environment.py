import os


def get_default_environment():
    return {
        "train": {
            "devices": "0,",
            "threads": 4,
            "workers": os.cpu_count() * 3 // 2 if os.cpu_count() is not None else 16,
        },
        "test": {
            "concurrency": os.cpu_count(),
        },
    }
