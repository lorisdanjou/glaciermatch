import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from scipy.ndimage import shift as ndshift

from conftest import make_texture, write_tif
from glaciermatch.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait_for(app, cond, timeout=20):
    t0 = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t0 < timeout, "timed out"


def test_full_workflow(app, tmp_path):
    base = make_texture((420, 480), seed=11)
    moved = ndshift(base, (-2.0, 6.5), order=3, mode="reflect")   # dy=-2, dx=+6.5
    p1 = write_tif(tmp_path / "a.tif", base[:, :400], (500000.0, 5000000.0))
    p2 = write_tif(tmp_path / "b.tif", moved[:, 40:], (500000.0 + 40 * 10, 5000000.0))

    w = MainWindow()
    w.show()
    w.open_image(1, p1)
    assert not w.btn_coreg.isEnabled()          # image 2 not chosen yet
    w.open_image(2, p2)
    assert w.btn_coreg.isEnabled() and w.checks_list.count() > 0

    w.run_coregistration()
    wait_for(app, lambda: w.coreg is not None)
    assert w.coreg.shape == (420, 360)
    assert w.view1.has_image() and w.view2.has_image()

    # click two points (scene coords == pixel coords) + one outside the window margin
    w._on_click(150.3, 200.7, Qt.LeftButton)
    w._on_click(250.5, 120.5, Qt.LeftButton)
    w._on_click(3.0, 3.0, Qt.LeftButton)
    assert w.points == [(150, 200), (250, 120)]
    w._on_click(150.5, 200.5, Qt.RightButton)   # remove first
    assert w.points == [(250, 120)]
    w._on_click(150.3, 200.7, Qt.LeftButton)

    w.params.sample.setValue(32)                 # forced odd
    assert w.params.sample.value() == 33
    assert w.params.search.value() > w.params.sample.value()
    w.params.dt.setValue(10.0)

    w.run_matching()
    wait_for(app, lambda: w.results is not None)
    assert len(w.results) == 2
    for r in w.results:
        assert r.ok, r.flags
        assert abs(r.dx - 6.5) < 0.2 and abs(r.dy + 2.0) < 0.2
    assert w.table["vel_m_day"].iloc[0] == pytest.approx(np.hypot(6.5, 2.0) * 10 / 10, abs=0.1)
    assert w.tbl.rowCount() == 2
    assert w.tbl.item(0, 3).text() == f"{w.results[0].dx:+.2f}"
    assert w.tbl.item(0, 9).text() == "ok"

    csv, gj = tmp_path / "o.csv", tmp_path / "o.geojson"
    w.export_csv(str(csv)); w.export_geojson(str(gj))
    assert csv.read_text().startswith("id,col,row")
    fc = json.loads(gj.read_text())
    assert len(fc["features"]) == 2 and fc["crs"]["properties"]["name"].endswith("32633")

    w.grab().save(str(tmp_path / "shot.png"))
    w.close()


def test_no_overlap_shows_error(app, tmp_path):
    arr = make_texture((100, 100), seed=1)
    p1 = write_tif(tmp_path / "a.tif", arr, (500000.0, 5000000.0))
    p2 = write_tif(tmp_path / "b.tif", arr, (600000.0, 5100000.0))
    w = MainWindow()
    w.open_image(1, p1); w.open_image(2, p2)
    assert not w.btn_coreg.isEnabled()
    assert any("do not overlap" in w.checks_list.item(i).text() for i in range(w.checks_list.count()))


def test_zoom_keeps_point_under_cursor(app, tmp_path):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent

    arr = make_texture((300, 400), seed=2)
    p1 = write_tif(tmp_path / "a.tif", arr, (500000.0, 5000000.0))
    w = MainWindow(); w.resize(1400, 800); w.show()
    w.open_image(1, p1); w.open_image(2, p1)
    w.run_coregistration()
    wait_for(app, lambda: w.coreg is not None)
    v = w.view1
    for pos in (QPoint(150, 120), QPoint(v.viewport().width() - 60, v.viewport().height() - 50)):
        for delta in (120, 120, 120, -120, 120):
            before = v.mapToScene(pos)
            ev = QWheelEvent(QPointF(pos), QPointF(v.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, delta),
                             Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
            v.wheelEvent(ev)
            after = v.mapToScene(pos)
            tol = 2.0 / v.transform().m11()   # integer scrollbars: within ~2 screen pixels
            assert abs(after.x() - before.x()) < tol and abs(after.y() - before.y()) < tol
    assert v.transform().m11() > 1.5
    # linked view follows the zoom level
    assert w.view2.transform().m11() == pytest.approx(v.transform().m11())
    w.close()
