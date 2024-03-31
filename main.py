import sys

from PyQt5.QtWidgets import QApplication

from MVC.Controller import Controller
from MVC.Model import Model
from MVC.View import View

if __name__ == '__main__':
    app = QApplication(sys.argv)

    view = View()
    model = Model()

    controller = Controller(model, view)
    sys.exit(app.exec_())