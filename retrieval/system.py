"""
retrieval/system.py
---------------------
Top-K video retrieval backed by the MultiTaskPINN -- FILE-LEVEL inference.
Each query pressure trace is treated as one whole file: resampled once to
SERIES_LENGTH for the TCN branch, with the hand-crafted features extracted
once from the full raw trace. There is no sliding-window slicing and no
aggregation across windows, since the model consumes (and was trained on)
one fixed-size vector per file.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..pinn.config import CLASS_NAMES, SUB_FOLDERS, DEVICE, MIN_TRACE_LEN
from ..pinn.models import MultiTaskPINN
from ..pinn.features import extract_pressure_features, resample_series
from ..pinn.splits import extract_velocities_from_filename
from .types import VideoEntry, RetrievalResult

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


class VideoRetrievalSystem:
    """
    Parameters
    ----------
    video_base_dir  : Root directory containing Dispersed-Flow/, Plug-Flow/,
                      Slug-Flow/ sub-folders with video files.
    model_ckpt_path : Path to best_pinn_model.pth  (state-dict only).
    scalers_path    : Path to best_scalar.pth  (dict with keys:
                      'pressure', 'features', 'vsg', 'vsl').
    cross_regime    : If True, Top-K search is performed across ALL regimes
                      (not just the predicted one). Default: False.
    """

    def __init__(
        self,
        video_base_dir:  str,
        model_ckpt_path: str,
        scalers_path:    str,
        cross_regime:    bool = False,
    ):
        self.video_base_dir = Path(video_base_dir)
        self.cross_regime   = cross_regime

        print(f"[Init] Device : {DEVICE}")

        # 1. Load scalers
        print(f"[Init] Loading scalers  → {scalers_path}")
        with open(scalers_path, "rb") as fh:
            self.scalers: dict = pickle.load(fh)
        self._validate_scalers()

        # 2. Load model
        print(f"[Init] Loading model    → {model_ckpt_path}")
        self.model = MultiTaskPINN().to(DEVICE)
        state = torch.load(model_ckpt_path, map_location=DEVICE)
        # Handle both raw state-dict and checkpoint-dict formats
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[Init] Model loaded successfully.")

        # 3. Build video index
        print(f"[Init] Indexing videos  → {self.video_base_dir}")
        self.index: List[VideoEntry] = self._build_index()

        # Pre-compute numpy arrays for fast distance calculation
        self._index_vsg = np.array([e.vsg for e in self.index], dtype=np.float32)
        self._index_vsl = np.array([e.vsl for e in self.index], dtype=np.float32)
        self._index_regime = np.array([e.regime_idx for e in self.index], dtype=np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        pressure_series: np.ndarray,
        k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Run FILE-LEVEL inference on a raw pressure time-series (the whole
        file's trace, any length) and return the Top-K most similar videos
        ranked by Euclidean distance on (Vsg, Vsl).

        Parameters
        ----------
        pressure_series : 1-D float array of raw pressure values (barA)
                          for the ENTIRE file -- no minimum length beyond
                          MIN_TRACE_LEN samples, since the whole trace is
                          resampled to SERIES_LENGTH internally.
        k               : Number of results to return.

        Returns
        -------
        List[RetrievalResult] sorted by ascending distance (best match first).
        """
        pressure_series = np.asarray(pressure_series, dtype=np.float32)
        if len(pressure_series) < MIN_TRACE_LEN:
            raise ValueError(
                f"Pressure series length {len(pressure_series)} is shorter than "
                f"MIN_TRACE_LEN={MIN_TRACE_LEN}. Cannot run inference."
            )

        # ── Step 1: Inference (file-level, one forward pass) ───────────
        regime_idx, vsg_pred, vsl_pred, regime_probs = self._infer(pressure_series)
        regime_name = CLASS_NAMES[regime_idx]

        print(f"[Inference]  Regime  : {regime_name}  (confidence {regime_probs[regime_idx]:.1%})")
        print(f"[Inference]  Vsg     : {vsg_pred:.4f} m/s")
        print(f"[Inference]  Vsl     : {vsl_pred:.4f} m/s")

        # ── Step 2: Filter candidate pool ─────────────────────────────
        if self.cross_regime:
            candidate_mask = np.ones(len(self.index), dtype=bool)
        else:
            candidate_mask = self._index_regime == regime_idx

        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            print(f"[Warning] No videos found for regime '{regime_name}'. "
                  f"Falling back to cross-regime search.")
            candidate_indices = np.arange(len(self.index))

        # ── Step 3: Euclidean distance on (Vsg, Vsl) ──────────────────
        cand_vsg = self._index_vsg[candidate_indices]
        cand_vsl = self._index_vsl[candidate_indices]

        distances = np.sqrt(
            (cand_vsg - vsg_pred) ** 2 +
            (cand_vsl - vsl_pred) ** 2
        )

        # ── Step 4: Top-K selection ────────────────────────────────────
        k_actual = min(k, len(candidate_indices))
        top_k_local = np.argsort(distances)[:k_actual]
        top_k_global = candidate_indices[top_k_local]

        results = []
        for rank, (global_idx, local_idx) in enumerate(
            zip(top_k_global, top_k_local), start=1
        ):
            entry = self.index[global_idx]
            results.append(RetrievalResult(
                rank           = rank,
                video_path     = entry.path,
                video_filename = entry.filename,
                regime_name    = entry.regime_name,
                vsg            = entry.vsg,
                vsl            = entry.vsl,
                distance       = float(distances[local_idx]),
                query_vsg      = vsg_pred,
                query_vsl      = vsl_pred,
                query_regime   = regime_name,
            ))

        return results

    def print_results(self, results: List[RetrievalResult]) -> None:
        """Pretty-print a retrieval result list to stdout."""
        if not results:
            print("[Results] No results to display.")
            return

        q = results[0]
        header = (
            f"\n{'='*70}\n"
            f"  Query  →  Regime: {q.query_regime}   "
            f"Vsg={q.query_vsg:.4f} m/s   Vsl={q.query_vsl:.4f} m/s\n"
            f"{'='*70}\n"
            f"  {'Rank':<5} {'Regime':<18} {'Vsg':>8} {'Vsl':>8} "
            f"{'Distance':>10}   Filename\n"
            f"  {'-'*65}"
        )
        print(header)
        for r in results:
            print(
                f"  {r.rank:<5} {r.regime_name:<18} "
                f"{r.vsg:>8.4f} {r.vsl:>8.4f} "
                f"{r.distance:>10.6f}   {r.video_filename}"
            )
        print(f"{'='*70}\n")

    def get_best_match(
        self,
        pressure_series: np.ndarray,
    ) -> Optional[RetrievalResult]:
        """Convenience wrapper — returns only the single best match."""
        results = self.retrieve(pressure_series, k=1)
        return results[0] if results else None

    def summary(self) -> None:
        """Print a summary of the video index."""
        print(f"\n{'='*50}")
        print(f"  VIDEO INDEX SUMMARY")
        print(f"{'='*50}")
        for idx, name in enumerate(CLASS_NAMES):
            count = sum(1 for e in self.index if e.regime_idx == idx)
            print(f"  {name:<20} : {count:>4} videos")
        print(f"  {'TOTAL':<20} : {len(self.index):>4} videos")
        print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_scalers(self) -> None:
        required = {"pressure", "features", "vsg", "vsl"}
        missing  = required - set(self.scalers.keys())
        if missing:
            raise KeyError(
                f"Scalers file is missing keys: {missing}. "
                f"Expected keys: {required}."
            )

    def _build_index(self) -> List[VideoEntry]:
        """
        Walk video_base_dir/SUB_FOLDERS and build a list of VideoEntry
        objects for every video file whose filename contains Vsg=X and Vsl=X.
        Files missing velocity info are skipped with a warning.
        """
        index = []
        skipped = 0

        for regime_idx, sub_folder in enumerate(SUB_FOLDERS):
            folder_path = self.video_base_dir / sub_folder
            if not folder_path.is_dir():
                print(f"  [Warning] Directory not found: {folder_path} — skipping.")
                continue

            for fname in sorted(os.listdir(folder_path)):
                ext = Path(fname).suffix.lower()
                if ext not in VIDEO_EXTS:
                    continue

                vsg, vsl = extract_velocities_from_filename(fname)
                if vsg is None or vsl is None:
                    print(f"  [Warning] Cannot parse Vsg/Vsl from '{fname}' — skipping.")
                    skipped += 1
                    continue

                index.append(VideoEntry(
                    path        = str(folder_path / fname),
                    filename    = fname,
                    regime_idx  = regime_idx,
                    regime_name = CLASS_NAMES[regime_idx],
                    vsg         = vsg,
                    vsl         = vsl,
                ))

        if skipped:
            print(f"  [Warning] Skipped {skipped} file(s) with unparseable filenames.")

        return index

    def _infer(self, pressure_series: np.ndarray):
        """
        FILE-LEVEL inference: resample the ENTIRE pressure trace to
        SERIES_LENGTH for the TCN branch, extract the hand-crafted
        features from the FULL raw trace, and run a single forward pass.
        No sliding window, no per-window aggregation.

        Returns:
            regime_idx  : int
            vsg_pred    : float  (original scale, m/s)
            vsl_pred    : float  (original scale, m/s)
            regime_probs: np.ndarray  shape (NUM_CLASSES,)
        """
        pressure_series = np.asarray(pressure_series, dtype=np.float32)

        feats = extract_pressure_features(pressure_series)
        pressure_fixed = resample_series(pressure_series)

        p_scaled = self.scalers["pressure"].transform(pressure_fixed.reshape(1, -1))
        f_scaled = self.scalers["features"].transform(feats.reshape(1, -1))

        p_tensor = torch.tensor(p_scaled, dtype=torch.float32).to(DEVICE)
        f_tensor = torch.tensor(f_scaled, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            cls_logits, vel_pred = self.model(p_tensor, f_tensor)

        logits_arr = cls_logits.cpu().numpy()  # (1, 3)
        softmax = np.exp(logits_arr) / np.exp(logits_arr).sum(axis=1, keepdims=True)
        regime_probs = softmax[0]  # (3,)
        regime_idx = int(np.argmax(regime_probs))

        vsg_scaled = vel_pred[:, 0:1].cpu().numpy()  # (1, 1)
        vsl_scaled = vel_pred[:, 1:2].cpu().numpy()  # (1, 1)

        vsg_pred = float(self.scalers["vsg"].inverse_transform(vsg_scaled)[0, 0])
        vsl_pred = float(self.scalers["vsl"].inverse_transform(vsl_scaled)[0, 0])

        return regime_idx, vsg_pred, vsl_pred, regime_probs
