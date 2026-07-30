"""
retrieval/types.py
--------------------
Plain data containers passed between VideoRetrievalSystem and RetrievalPlayer.
"""

from dataclasses import dataclass


@dataclass
class VideoEntry:
    """One record in the video index."""
    path:        str
    filename:    str
    regime_idx:  int
    regime_name: str
    vsg:         float
    vsl:         float


@dataclass
class RetrievalResult:
    """One Top-K hit returned to the caller."""
    rank:             int
    video_path:       str
    video_filename:   str
    regime_name:      str
    vsg:              float
    vsl:              float
    distance:         float  # Euclidean distance in (Vsg, Vsl) space
    query_vsg:        float
    query_vsl:        float
    query_regime:     str
