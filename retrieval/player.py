"""
retrieval/player.py
---------------------
Interactive Jupyter widget that lets you browse Top-K retrieval results
and play each matched video inline.
"""

import os

import ipywidgets as widgets
from IPython.display import display, HTML

from .display import REGIME_COLORS, badge, video_html


class RetrievalPlayer:
    """
    Parameters
    ----------
    results  : List of RetrievalResult objects from VideoRetrievalSystem.retrieve()
    width    : Video display width in pixels (default 640).
    autoplay : Whether the video starts playing automatically (default True).

    Example
    -------
    results = vrs.retrieve(pressure_series, k=5)
    player  = RetrievalPlayer(results)
    player.show()
    """

    def __init__(self, results, width: int = 640, autoplay: bool = True):
        if not results:
            raise ValueError("results list is empty — nothing to play.")

        self.results  = results
        self.width    = width
        self.autoplay = autoplay
        self._current = 0  # currently displayed result index

        self._build_ui()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def show(self):
        """Render the player widget in the current Jupyter cell."""
        display(self._root)
        self._render(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        n = len(self.results)

        # ── Header ────────────────────────────────────────────────────
        q = self.results[0]
        header_html = (
            f'<div style="font-family:monospace;padding:8px 0;">'
            f'<b>Query</b> &nbsp;→&nbsp; '
            f'{badge(q.query_regime, REGIME_COLORS.get(q.query_regime,"#555"))} &nbsp; '
            f'Vsg&nbsp;=&nbsp;<b>{q.query_vsg:.4f}</b>&nbsp;m/s &nbsp; '
            f'Vsl&nbsp;=&nbsp;<b>{q.query_vsl:.4f}</b>&nbsp;m/s'
            f'</div>'
        )
        self._header = widgets.HTML(value=header_html)

        # ── Navigation buttons ─────────────────────────────────────────
        btn_style = dict(button_style="", layout=widgets.Layout(width="110px"))

        self._btn_prev = widgets.Button(description="◀  Prev", **btn_style)
        self._btn_next = widgets.Button(description="Next  ▶", **btn_style)
        self._rank_label = widgets.HTML()

        self._btn_prev.on_click(lambda _: self._navigate(-1))
        self._btn_next.on_click(lambda _: self._navigate(+1))

        nav_bar = widgets.HBox(
            [self._btn_prev, self._rank_label, self._btn_next],
            layout=widgets.Layout(align_items="center", gap="12px", margin="4px 0"),
        )

        # ── Result metadata panel ──────────────────────────────────────
        self._meta_panel = widgets.HTML()

        # ── Video output area ──────────────────────────────────────────
        self._video_out = widgets.Output(
            layout=widgets.Layout(margin="6px 0")
        )

        # ── Results table (collapsed by default) ──────────────────────
        self._table_out  = widgets.Output()
        self._table_acc  = widgets.Accordion(children=[self._table_out])
        self._table_acc.set_title(0, f"📋  All {n} results")
        self._table_acc.selected_index = None  # collapsed

        self._render_table()

        # ── Root layout ────────────────────────────────────────────────
        self._root = widgets.VBox(
            [
                self._header,
                widgets.HTML("<hr style='margin:4px 0;border-color:#ddd;'>"),
                nav_bar,
                self._meta_panel,
                self._video_out,
                self._table_acc,
            ],
            layout=widgets.Layout(
                padding="10px",
                border="1px solid #ddd",
                border_radius="8px",
                max_width="700px",
            ),
        )

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _navigate(self, delta: int):
        new_idx = (self._current + delta) % len(self.results)
        self._render(new_idx)

    def _render(self, idx: int):
        self._current = idx
        r = self.results[idx]
        n = len(self.results)

        # Rank label
        self._rank_label.value = (
            f'<span style="font-size:14px;font-weight:600;">'
            f'Result {idx+1} / {n}</span>'
        )

        # Prev / Next button states
        self._btn_prev.disabled = (n == 1)
        self._btn_next.disabled = (n == 1)

        # Metadata panel
        regime_color = REGIME_COLORS.get(r.regime_name, "#555")
        self._meta_panel.value = f"""
        <div style="font-family:monospace;font-size:13px;
                    background:#f8f8f8;border-radius:6px;padding:8px 12px;
                    line-height:1.8;">
            <b>Rank</b>         &nbsp;→&nbsp; {r.rank} &nbsp;
            {badge(r.regime_name, regime_color)}<br>
            <b>Vsg</b>          &nbsp;→&nbsp; {r.vsg:.4f} m/s
            &nbsp;&nbsp;
            <b>Vsl</b>          &nbsp;→&nbsp; {r.vsl:.4f} m/s<br>
            <b>Distance</b>     &nbsp;→&nbsp; {r.distance:.6f}<br>
            <b>File</b>         &nbsp;→&nbsp;
            <span style="color:#555;">{r.video_filename}</span>
        </div>
        """

        # Video
        self._video_out.clear_output(wait=True)
        with self._video_out:
            if not os.path.isfile(r.video_path):
                display(HTML(
                    f'<p style="color:red;font-family:monospace;">'
                    f'⚠ File not found:<br>{r.video_path}</p>'
                ))
            else:
                try:
                    display(HTML(video_html(r.video_path, width=self.width)))
                except Exception as exc:
                    display(HTML(
                        f'<p style="color:red;font-family:monospace;">'
                        f'⚠ Could not load video: {exc}</p>'
                    ))

    def _render_table(self):
        """Render the full results table inside the accordion."""
        rows = ""
        for r in self.results:
            color  = REGIME_COLORS.get(r.regime_name, "#555")
            is_cur = r.rank == 1
            bg     = "#fffbe6" if is_cur else "#fff"
            rows += (
                f'<tr style="background:{bg};">'
                f'<td style="text-align:center;">{r.rank}</td>'
                f'<td><span style="background:{color};color:#fff;'
                f'padding:1px 6px;border-radius:3px;font-size:11px;">'
                f'{r.regime_name}</span></td>'
                f'<td style="text-align:right;">{r.vsg:.4f}</td>'
                f'<td style="text-align:right;">{r.vsl:.4f}</td>'
                f'<td style="text-align:right;">{r.distance:.6f}</td>'
                f'<td style="font-size:11px;color:#555;">{r.video_filename}</td>'
                f'</tr>'
            )

        table_html = f"""
        <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;
                      font-family:monospace;font-size:12px;">
            <thead>
                <tr style="background:#4a90d9;color:#fff;">
                    <th style="padding:5px 8px;">Rank</th>
                    <th style="padding:5px 8px;">Regime</th>
                    <th style="padding:5px 8px;">Vsg (m/s)</th>
                    <th style="padding:5px 8px;">Vsl (m/s)</th>
                    <th style="padding:5px 8px;">Distance</th>
                    <th style="padding:5px 8px;">Filename</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        """
        with self._table_out:
            display(HTML(table_html))
