"""Launch the glaciermatch GUI: python main.py [image1.tif image2.tif]"""
import sys

from PySide6.QtWidgets import QApplication

from glaciermatch.gui.main_window import MainWindow


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    win = MainWindow()
    win.show()
    if len(argv) == 3:
        win.open_image(1, argv[1])
        win.open_image(2, argv[2])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
