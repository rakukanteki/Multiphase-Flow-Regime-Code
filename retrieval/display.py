"""
retrieval/display.py
----------------------
Small HTML/base64 helpers used to embed a matched video and styled badges
inline in a Jupyter cell.
"""

import base64
from pathlib import Path

REGIME_COLORS = {
    "Dispersed Flow": "#2ecc71",
    "Plug Flow":      "#e67e22",
    "Slug Flow":      "#9b59b6",
}


def video_to_b64(path: str) -> str:
    """Read a video file from disk and return a base64-encoded string."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".mp4":  "video/mp4",
        ".avi":  "video/x-msvideo",
        ".mov":  "video/quicktime",
        ".mkv":  "video/x-matroska",
        ".wmv":  "video/x-ms-wmv",
        ".webm": "video/webm",
    }.get(ext, "video/mp4")


def video_html(path: str, width: int = 640) -> str:
    """Return an HTML5 <video> tag with the file embedded as base64."""
    b64  = video_to_b64(path)
    mime = mime_type(path)
    return f"""
    <video width="{width}" controls autoplay loop
           style="border-radius:8px; border:2px solid #4a90d9; margin-top:6px;">
        <source src="data:{mime};base64,{b64}" type="{mime}">
        Your browser does not support the HTML5 video tag.
    </video>
    """


def badge(text: str, color: str = "#4a90d9") -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:600;">{text}</span>'
    )
