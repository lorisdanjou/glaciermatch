"""Zoomable/pannable image view with overlay helpers (scene coords == image pixel coords)."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView,
)

_DRAG_PX = 4


def to_uint8(img: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> np.ndarray:
    """Percentile stretch for display; NaN -> 0."""
    ok = np.isfinite(img)
    if not ok.any():
        return np.zeros(img.shape, np.uint8)
    lo, hi = np.percentile(img[ok], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    out = np.clip((img - lo) / (hi - lo), 0, 1)
    out = np.where(ok, out, 0)
    return np.ascontiguousarray((out * 255).astype(np.uint8))


class ImageView(QGraphicsView):
    clicked = Signal(float, float, object)   # scene x, y, Qt.MouseButton
    mouseMoved = Signal(float, float)        # scene x, y
    viewChanged = Signal()

    def __init__(self, parent=None):
        self._scene = QGraphicsScene()   # keep a Python reference
        super().__init__(self._scene, parent)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)  # zoom anchoring is done in wheelEvent
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self._pix: QGraphicsPixmapItem | None = None
        self._img_rect = QRectF()
        self._qimage: QImage | None = None
        self._layers: dict[str, list[QGraphicsItem]] = {}
        self._press = None
        self._dragging = False
        self.horizontalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())

    # ---- image -------------------------------------------------------------
    def set_image(self, arr_u8: np.ndarray) -> None:
        h, w = arr_u8.shape
        self._qimage = QImage(arr_u8.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.clear_all_layers()
        self.scene().clear()
        self._pix = self.scene().addPixmap(QPixmap.fromImage(self._qimage))
        self._pix.setZValue(0)
        self._img_rect = QRectF(0, 0, w, h)
        # Margin around the image so the view can scroll freely: without it, an image smaller
        # than the viewport has no scroll range and zooming cannot keep the cursor point fixed.
        m = max(w, h)
        self.scene().setSceneRect(self._img_rect.adjusted(-m, -m, m, m))
        self.fit()

    def has_image(self) -> bool:
        return self._pix is not None

    def fit(self) -> None:
        if self._pix is not None:
            self.fitInView(self._img_rect, Qt.KeepAspectRatio)
            self.viewChanged.emit()

    def sync_from(self, other: "ImageView") -> None:
        if not (self.has_image() and other.has_image()):
            return
        c = other.mapToScene(other.viewport().rect().center())
        self.setTransform(other.transform())
        self.centerOn(c)

    # ---- overlays ----------------------------------------------------------
    def clear_layer(self, name: str) -> None:
        for it in self._layers.pop(name, []):
            self.scene().removeItem(it)

    def clear_all_layers(self) -> None:
        for name in list(self._layers):
            self._layers.pop(name)  # items are removed with scene.clear()

    def _add(self, layer: str, item: QGraphicsItem, z: float = 10) -> None:
        item.setZValue(z)
        self.scene().addItem(item)
        self._layers.setdefault(layer, []).append(item)

    def add_marker(self, layer, x, y, label="", color="#ffd400", size=6, highlighted=False):
        path = QPainterPath()
        path.moveTo(-size, 0); path.lineTo(size, 0)
        path.moveTo(0, -size); path.lineTo(0, size)
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(color), 3 if highlighted else 1.5)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setPos(x, y)
        item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        if label:
            t = QGraphicsSimpleTextItem(str(label), item)  # child: inherits the fixed screen size
            t.setBrush(QBrush(QColor(color)))
            t.setPos(size, -size - 14)
        self._add(layer, item)

    def add_rect(self, layer, cx, cy, size, color, dashed=False):
        r = QGraphicsRectItem(cx - size / 2, cy - size / 2, size, size)
        pen = QPen(QColor(color), 1.5)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.DashLine)
        r.setPen(pen)
        self._add(layer, r, z=5)

    def add_arrow(self, layer, x0, y0, x1, y1, color="#00e676"):
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        line = QGraphicsLineItem(x0, y0, x1, y1)
        line.setPen(pen)
        self._add(layer, line)
        if math.hypot(x1 - x0, y1 - y0) > 1e-9:
            head = QGraphicsPolygonItem(QPolygonF([QPointF(0, 0), QPointF(-9, 4), QPointF(-9, -4)]))
            head.setBrush(QBrush(QColor(color)))
            head.setPen(QPen(Qt.NoPen))
            head.setPos(x1, y1)
            head.setRotation(math.degrees(math.atan2(y1 - y0, x1 - x0)))
            head.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self._add(layer, head)

    # ---- interaction -------------------------------------------------------
    def wheelEvent(self, ev):
        if not self.has_image():
            return
        f = 1.25 ** (ev.angleDelta().y() / 120.0)
        s = self.transform().m11() * f
        if 0.02 <= s <= 300:
            # Keep the scene point under the cursor fixed on screen.
            pos = ev.position().toPoint()
            before = self.mapToScene(pos)
            self.scale(f, f)
            # With NoAnchor the scroll position is unchanged, so the view centre moved too.
            centre = self.mapToScene(self.viewport().rect().center())
            after = self.mapToScene(pos)
            self.centerOn(centre + before - after)
            self.viewChanged.emit()

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._press = (ev.button(), ev.position())
            self._last = ev.position()
            self._dragging = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        p = self.mapToScene(ev.position().toPoint())
        self.mouseMoved.emit(p.x(), p.y())
        if self._press is not None:
            if not self._dragging:
                d = ev.position() - self._press[1]
                if abs(d.x()) + abs(d.y()) > _DRAG_PX:
                    self._dragging = True
                    self.viewport().setCursor(Qt.ClosedHandCursor)
            if self._dragging:
                d = ev.position() - self._last
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
        self._last = ev.position()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if self._press is not None and ev.button() == self._press[0]:
            if not self._dragging and ev.button() in (Qt.LeftButton, Qt.RightButton) and self.has_image():
                p = self.mapToScene(ev.position().toPoint())
                self.clicked.emit(p.x(), p.y(), ev.button())
            self._press = None
            self._dragging = False
            self.viewport().unsetCursor()
        ev.accept()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self.fit()
