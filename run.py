"""Launch the desktop tracker: python run.py [video_path]."""
import argparse
import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = MainWindow()
    window.show()
    if args.video:
        window.load_video(args.video)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
