"""Matching parameters panel."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSpinBox, QVBoxLayout, QWidget,
)


class ParamsPanel(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample = QSpinBox(); self.sample.setRange(5, 501); self.sample.setSingleStep(2)
        self.sample.setValue(31); self.sample.setSuffix(" px")
        self.search = QSpinBox(); self.search.setRange(7, 1001); self.search.setSingleStep(2)
        self.search.setValue(71); self.search.setSuffix(" px")
        self.levels = QSpinBox(); self.levels.setRange(1, 8); self.levels.setValue(3)
        self.refine = QSpinBox(); self.refine.setRange(1, 20); self.refine.setValue(4)
        self.refine.setSuffix(" px")
        self.min_corr = QDoubleSpinBox(); self.min_corr.setRange(0, 1); self.min_corr.setSingleStep(0.05)
        self.min_corr.setValue(0.5)
        self.min_ratio = QDoubleSpinBox(); self.min_ratio.setRange(1, 10); self.min_ratio.setSingleStep(0.05)
        self.min_ratio.setValue(1.05)
        self.subpixel = QCheckBox("Sub-pixel refinement"); self.subpixel.setChecked(True)
        self.dt = QDoubleSpinBox(); self.dt.setRange(0, 1e5); self.dt.setDecimals(2)
        self.dt.setSuffix(" days"); self.dt.setSpecialValueText("unknown")
        self.arrow_scale = QDoubleSpinBox(); self.arrow_scale.setRange(0.1, 1000)
        self.arrow_scale.setValue(1.0); self.arrow_scale.setPrefix("× ")

        box = QGroupBox("Matching")
        f = QFormLayout(box)
        f.addRow("Sample window", self.sample)
        f.addRow("Search window", self.search)
        f.addRow("Pyramid levels", self.levels)
        f.addRow("Refine radius", self.refine)
        f.addRow("Min. correlation", self.min_corr)
        f.addRow("Min. peak ratio", self.min_ratio)
        f.addRow(self.subpixel)
        box2 = QGroupBox("Output")
        f2 = QFormLayout(box2)
        f2.addRow("Time interval", self.dt)
        f2.addRow("Arrow scale", self.arrow_scale)
        lay = QVBoxLayout(self)
        lay.addWidget(box)
        lay.addWidget(box2)

        self.sample.valueChanged.connect(self._fix_sizes)
        self.search.valueChanged.connect(self._fix_sizes)
        for w in (self.levels, self.refine, self.min_corr, self.min_ratio, self.dt, self.arrow_scale):
            w.valueChanged.connect(lambda *_: self.changed.emit())
        self.subpixel.toggled.connect(lambda *_: self.changed.emit())

    def _fix_sizes(self, *_):
        """Keep both sizes odd and the search window strictly larger than the sample."""
        for w in (self.sample, self.search):
            if w.value() % 2 == 0:
                w.blockSignals(True); w.setValue(w.value() + 1); w.blockSignals(False)
        if self.search.value() <= self.sample.value():
            self.search.blockSignals(True)
            self.search.setValue(self.sample.value() + 2)
            self.search.blockSignals(False)
        self.changed.emit()

    def values(self) -> dict:
        return dict(
            sample_size=self.sample.value(),
            search_size=self.search.value(),
            n_levels=self.levels.value(),
            refine_radius=self.refine.value(),
            min_corr=self.min_corr.value(),
            min_peak_ratio=self.min_ratio.value(),
            subpixel=self.subpixel.isChecked(),
        )

    @property
    def dt_days(self) -> float | None:
        v = self.dt.value()
        return v if v > 0 else None
