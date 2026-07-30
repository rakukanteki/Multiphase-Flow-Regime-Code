"""
retrieval/demo.py
-------------------
Runnable demo mirroring the notebook's bottom cell: build the retrieval
system from the trained checkpoint + scalers, run a Top-5 query, print the
results, and launch the interactive player.

Usage:
    python -m retrieval.demo
"""

import numpy as np

from .system import VideoRetrievalSystem
from .player import RetrievalPlayer

VIDEO_BASE_DIR  = "./Videos"
MODEL_CKPT_PATH = "models/best_pinn_model.pth"
SCALERS_PATH    = "models/best_scalar.pth"


def main():
    vrs = VideoRetrievalSystem(
        video_base_dir  = VIDEO_BASE_DIR,
        model_ckpt_path = MODEL_CKPT_PATH,
        scalers_path    = SCALERS_PATH,
    )

    print("[Demo] Running Top-5 retrieval on a pressure signal ...\n")
    demo_pressure = np.random.randn(200).astype(np.float32) * 0.01 + 1.5

    results = vrs.retrieve(pressure_series=demo_pressure, k=5)
    vrs.print_results(results)

    best = results[0]
    print(f"Best match path   : {best.video_path}")
    print(f"Best match regime : {best.regime_name}")
    print(f"Best match Vsg    : {best.vsg:.4f} m/s")
    print(f"Best match Vsl    : {best.vsl:.4f} m/s")
    print(f"Distance          : {best.distance:.6f}")

    player = RetrievalPlayer(results, width=640)
    player.show()


if __name__ == "__main__":
    main()
