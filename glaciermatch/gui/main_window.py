"""Main window: two linked views, parameters, check summary, results table."""
from __future__ import annotations

import math
import os

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..checks import Status, check_inputs, has_errors
from ..coregistration import CoregistrationError, coregister
from ..io import read_info
from ..multiscale import match_points
from ..results import build_table, export_csv, export_geojson
from .image_view import ImageView, to_uint8
from .params_panel import ParamsPanel

_STATUS_STYLE = {
    Status.OK: ("✔", "#2e7d32"),
    Status.WARN: ("⚠", "#e65100"),
    Status.ERROR: ("✖", "#c62828"),
}
_PICK_RADIUS_PX = 10   # screen pixels within which a click selects an existing point
_COLOR_PT, _COLOR_SEL = "#ffd400", "#ff4081"
_COLOR_OK, _COLOR_BAD = "#00e676", "#ff9100"
TABLE_HEADERS = ["#", "col", "row", "dx px", "dy px", "|d| m", "v m/day", "corr", "ratio", "status"]


class Worker(QThread):
    """Runs ``fn(worker)`` off the GUI thread. ``fn`` may call ``worker.report(i, n)``."""
    done = Signal(object)
    failed = Signal(object)
    progress = Signal(int, int)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def report(self, i: int, n: int) -> None:
        self.progress.emit(i, n)
        if self.isInterruptionRequested():
            raise InterruptedError

    def run(self):
        try:
            self.done.emit(self._fn(self))
        except InterruptedError:
            self.failed.emit(None)
        except Exception as e:  # reported to the user, not swallowed
            self.failed.emit(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("glaciermatch")
        self.resize(1500, 900)
        self.path1 = self.path2 = None
        self.coreg = None
        self.points: list[tuple[int, int]] = []
        self.results = None
        self.table = None           # DataFrame of the last run
        self.selected: int | None = None
        self._worker: Worker | None = None
        self._syncing = False

        self.view1, self.view2 = ImageView(), ImageView()
        split = QSplitter()
        for title, v in (("Image 1 (click to add points)", self.view1), ("Image 2", self.view2)):
            w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(QLabel(f"<b>{title}</b>")); l.addWidget(v)
            split.addWidget(w)
        split.setSizes([1, 1])
        self.setCentralWidget(split)
        self.view1.clicked.connect(self._on_click)
        self.view1.mouseMoved.connect(self._on_mouse)
        self.view2.mouseMoved.connect(self._on_mouse)
        self.view1.viewChanged.connect(lambda: self._sync(self.view1, self.view2))
        self.view2.viewChanged.connect(lambda: self._sync(self.view2, self.view1))

        self._build_side_dock()
        self._build_table_dock()
        self._build_menu()
        self.resizeDocks([self.tbl_dock], [220], Qt.Vertical)
        self.statusBar().showMessage("Open image 1 and image 2 (File menu).")

    # ---- construction ------------------------------------------------------
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        def act(text, fn, shortcut=None):
            a = QAction(text, self); a.triggered.connect(fn)
            if shortcut: a.setShortcut(shortcut)
            m.addAction(a); return a
        act("Open image &1…", lambda: self.open_image(1), "Ctrl+1")
        act("Open image &2…", lambda: self.open_image(2), "Ctrl+2")
        m.addSeparator()
        self.act_export_csv = act("Export &CSV…", self.export_csv)
        self.act_export_geojson = act("Export &GeoJSON…", self.export_geojson)
        m.addSeparator()
        act("&Quit", self.close, QKeySequence.Quit)
        v = self.menuBar().addMenu("&View")
        a = QAction("&Fit both views", self); a.setShortcut("F"); a.triggered.connect(self._fit); v.addAction(a)
        p = self.menuBar().addMenu("&Points")
        a = QAction("&Remove selected", self); a.setShortcut("Delete"); a.triggered.connect(self.remove_selected); p.addAction(a)
        a = QAction("&Clear all", self); a.triggered.connect(self.clear_points); p.addAction(a)
        self._set_exports_enabled(False)

    def _build_side_dock(self):
        w = QWidget(); l = QVBoxLayout(w)
        self.lbl_files = QLabel("Image 1: –\nImage 2: –"); self.lbl_files.setWordWrap(True)
        l.addWidget(self.lbl_files)
        self.btn_coreg = QPushButton("Co-register"); self.btn_coreg.setEnabled(False)
        self.btn_coreg.clicked.connect(self.run_coregistration)
        l.addWidget(self.btn_coreg)
        self.checks_list = QListWidget(); self.checks_list.setWordWrap(True)
        self.checks_list.setMinimumHeight(160)
        l.addWidget(QLabel("<b>Input checks</b>")); l.addWidget(self.checks_list)
        self.params = ParamsPanel(); l.addWidget(self.params)
        self.params.changed.connect(self._on_params_changed)
        self.chk_link = QCheckBox("Link zoom and pan"); self.chk_link.setChecked(True)
        l.addWidget(self.chk_link)
        row = QHBoxLayout()
        self.btn_run = QPushButton("Run matching"); self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_matching)
        self.btn_cancel = QPushButton("Cancel"); self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)
        row.addWidget(self.btn_run); row.addWidget(self.btn_cancel)
        l.addLayout(row)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        l.addWidget(self.progress)
        l.addStretch()
        d = QDockWidget("Controls", self); d.setWidget(w)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, d)

    def _build_table_dock(self):
        self.tbl = QTableWidget(0, len(TABLE_HEADERS))
        self.tbl.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.tbl.setMinimumHeight(180)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl.setSelectionMode(QTableWidget.SingleSelection)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.itemSelectionChanged.connect(self._on_table_select)
        d = self.tbl_dock = QDockWidget("Points / results", self); d.setWidget(self.tbl)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)

    # ---- loading + co-registration ----------------------------------------
    def open_image(self, which: int, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, f"Open image {which}", "", "GeoTIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        if which == 1:
            self.path1 = path
        else:
            self.path2 = path
        self.lbl_files.setText(
            f"Image 1: {os.path.basename(self.path1) if self.path1 else '–'}\n"
            f"Image 2: {os.path.basename(self.path2) if self.path2 else '–'}")
        self._reset_coreg()
        if self.path1 and self.path2:
            self._pre_check()

    def _reset_coreg(self):
        self.coreg = None
        self.points.clear(); self.results = None; self.table = None; self.selected = None
        self.btn_run.setEnabled(False)
        self._set_exports_enabled(False)
        self._refresh_table()

    def _show_checks(self, checks):
        self.checks_list.clear()
        for c in checks:
            sym, col = _STATUS_STYLE[c.status]
            it = QListWidgetItem(f"{sym} {c.message}")
            it.setForeground(QColor(col))
            self.checks_list.addItem(it)

    def _pre_check(self):
        try:
            checks = check_inputs(read_info(self.path1), read_info(self.path2))
        except Exception as e:
            QMessageBox.critical(self, "Cannot read images", str(e))
            return
        self._show_checks(checks)
        self.btn_coreg.setEnabled(not has_errors(checks))
        self.statusBar().showMessage(
            "Checks failed – fix the inputs." if has_errors(checks) else "Checks passed. Click 'Co-register'.")

    def run_coregistration(self):
        self.btn_coreg.setEnabled(False)
        self.statusBar().showMessage("Co-registering…")
        p1, p2 = self.path1, self.path2
        self._start(lambda w: coregister(p1, p2), self._coreg_done, indeterminate=True)

    def _coreg_done(self, res):
        self.coreg = res
        self._show_checks(res.checks)
        self.view1.set_image(to_uint8(res.img1))
        self.view2.set_image(to_uint8(res.img2))
        h, w = res.shape
        self.statusBar().showMessage(
            f"Co-registered: {w}×{h} px, pixel size {res.pixel_size:g}, image 2 {res.resampling}. "
            "Left-click image 1 to add points, right-click to remove.")
        self.btn_coreg.setEnabled(True)
        self._refresh_all()

    # ---- threading helper --------------------------------------------------
    def _start(self, fn, on_done, indeterminate=False):
        if self._worker is not None and self._worker.isRunning():
            return
        wk = Worker(fn, self)
        self._worker = wk
        self.progress.setVisible(True)
        self.progress.setRange(0, 0 if indeterminate else 100)
        self.btn_cancel.setEnabled(not indeterminate)
        wk.progress.connect(lambda i, n: (self.progress.setMaximum(n), self.progress.setValue(i)))
        wk.done.connect(lambda r: (self._worker_finished(), on_done(r)))
        wk.failed.connect(self._worker_failed)
        wk.start()

    def _worker_finished(self):
        self.progress.setVisible(False)
        self.btn_cancel.setEnabled(False)
        self._update_run_enabled()

    def _worker_failed(self, err):
        self._worker_finished()
        self.btn_coreg.setEnabled(bool(self.path1 and self.path2))
        if err is None:
            self.statusBar().showMessage("Cancelled.")
            return
        if isinstance(err, CoregistrationError):
            self._show_checks(err.checks)
        QMessageBox.critical(self, "Error", f"{type(err).__name__}: {err}")
        self.statusBar().showMessage("Failed.")

    def cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()

    # ---- points ------------------------------------------------------------
    def _on_click(self, sx: float, sy: float, button):
        if self.coreg is None:
            return
        scale = self.view1.transform().m11()
        near = self._nearest_point(sx, sy, _PICK_RADIUS_PX / scale)
        if button == Qt.RightButton:
            if near is not None:
                self._remove(near)
            return
        if near is not None:
            self.selected = near
            self._refresh_all()
            return
        col, row = int(math.floor(sx)), int(math.floor(sy))
        h, w = self.coreg.shape
        if not (0 <= col < w and 0 <= row < h) or not self.coreg.valid[row, col]:
            self.statusBar().showMessage("Point is outside the valid overlap.")
            return
        T = self.params.sample.value()
        half = T // 2
        if col - half < 0 or row - half < 0 or col + half >= w or row + half >= h:
            self.statusBar().showMessage("Sample window would extend outside the image; choose a point further in.")
            return
        self.points.append((col, row))
        self.results = None; self.table = None
        self.selected = len(self.points) - 1
        self._refresh_all()

    def _nearest_point(self, sx, sy, radius):
        best, bd = None, radius
        for i, (c, r) in enumerate(self.points):
            d = math.hypot(c + 0.5 - sx, r + 0.5 - sy)
            if d <= bd:
                best, bd = i, d
        return best

    def _remove(self, i: int):
        del self.points[i]
        self.results = None; self.table = None
        self.selected = None if not self.points else min(i, len(self.points) - 1)
        self._refresh_all()

    def remove_selected(self):
        if self.selected is not None:
            self._remove(self.selected)

    def clear_points(self):
        self.points.clear(); self.results = None; self.table = None; self.selected = None
        self._refresh_all()

    # ---- matching ----------------------------------------------------------
    def _update_run_enabled(self):
        busy = self._worker is not None and self._worker.isRunning()
        self.btn_run.setEnabled(self.coreg is not None and bool(self.points) and not busy)

    def run_matching(self):
        if self.coreg is None or not self.points:
            return
        a1, a2 = self.coreg.matching_arrays()
        valid, pts, kw = self.coreg.valid, list(self.points), self.params.values()
        self.btn_run.setEnabled(False)
        self.statusBar().showMessage(f"Matching {len(pts)} point(s)…")
        self._start(
            lambda w: match_points(a1, a2, pts, valid=valid, progress=w.report, **kw),
            self._matching_done)

    def _matching_done(self, results):
        self.results = results
        self.table = build_table(self.coreg, self.points, results, self.params.dt_days)
        n_ok = int(self.table["ok"].sum())
        self.statusBar().showMessage(f"Done: {n_ok}/{len(results)} matches passed all checks.")
        self._set_exports_enabled(True)
        self._refresh_all()

    # ---- display -----------------------------------------------------------
    def _refresh_all(self):
        self._refresh_table()
        self._refresh_overlays()
        self._update_run_enabled()

    def _refresh_table(self):
        t = self.tbl
        t.blockSignals(True)
        t.setRowCount(len(self.points))
        for i, (c, r) in enumerate(self.points):
            if self.table is not None and i < len(self.table):
                rec = self.table.iloc[i]
                vals = [i + 1, c, r, f"{rec['dx_px']:+.2f}", f"{rec['dy_px']:+.2f}", f"{rec['disp_m']:.1f}" if np.isfinite(rec["disp_m"]) else "–",
                        f"{rec['vel_m_day']:.3f}" if np.isfinite(rec["vel_m_day"]) else "–", f"{rec['corr']:.2f}", f"{rec['peak_ratio']:.2f}",
                        "ok" if rec["ok"] else rec["flags"]]
            else:
                vals = [i + 1, c, r] + [""] * 7
            for j, v in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(str(v)))
        if self.selected is not None and self.selected < len(self.points):
            t.selectRow(self.selected)
        t.blockSignals(False)

    def _on_table_select(self):
        rows = self.tbl.selectionModel().selectedRows()
        if rows:
            self.selected = rows[0].row()
            self._refresh_overlays()

    def _refresh_overlays(self):
        for v in (self.view1, self.view2):
            for layer in ("points", "windows", "results"):
                v.clear_layer(layer)
        if self.coreg is None:
            return
        for i, (c, r) in enumerate(self.points):
            sel = i == self.selected
            self.view1.add_marker("points", c + 0.5, r + 0.5, i + 1,
                                  _COLOR_SEL if sel else _COLOR_PT, highlighted=sel)
        if self.selected is not None and self.selected < len(self.points):
            c, r = self.points[self.selected]
            T, S = self.params.sample.value(), self.params.search.value()
            self.view1.add_rect("windows", c + 0.5, r + 0.5, T, _COLOR_SEL)
            self.view2.add_rect("windows", c + 0.5, r + 0.5, T, _COLOR_SEL)
            self.view2.add_rect("windows", c + 0.5, r + 0.5, S, "#40c4ff", dashed=True)
        if self.results is not None:
            k = self.params.arrow_scale.value()
            for i, ((c, r), m) in enumerate(zip(self.points, self.results)):
                if not np.isfinite(m.dx):
                    continue
                col = _COLOR_OK if m.ok else _COLOR_BAD
                x0, y0 = c + 0.5, r + 0.5
                self.view2.add_arrow("results", x0, y0, x0 + k * m.dx, y0 + k * m.dy, col)
                self.view2.add_marker("results", x0 + m.dx, y0 + m.dy, "", col, size=4)

    def _on_params_changed(self):
        self._refresh_overlays()

    # ---- misc --------------------------------------------------------------
    def _sync(self, src: ImageView, dst: ImageView):
        if self._syncing or not self.chk_link.isChecked():
            return
        self._syncing = True
        try:
            dst.sync_from(src)
        finally:
            self._syncing = False

    def _fit(self):
        self.view1.fit()
        self.view2.fit()

    def _on_mouse(self, sx, sy):
        if self.coreg is None:
            return
        x, y = self.coreg.pix_to_map(sx - 0.5, sy - 0.5)
        self.statusBar().showMessage(f"col {int(math.floor(sx))}, row {int(math.floor(sy))}   |   E {x:.1f}  N {y:.1f}")

    def _set_exports_enabled(self, on: bool):
        self.act_export_csv.setEnabled(on)
        self.act_export_geojson.setEnabled(on)

    def export_csv(self, path: str | None = None):
        if self.table is None:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "matches.csv", "CSV (*.csv)")
        if path:
            export_csv(self.table, path)
            self.statusBar().showMessage(f"Saved {path}")

    def export_geojson(self, path: str | None = None):
        if self.table is None:
            return
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export GeoJSON", "matches.geojson", "GeoJSON (*.geojson *.json)")
        if path:
            export_geojson(self.table, path, self.coreg.crs)
            self.statusBar().showMessage(f"Saved {path}")

    def closeEvent(self, ev):
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(5000)
        super().closeEvent(ev)
