import sys

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog

import GlobalVariables
from MvcView import View
from MvcModel import Model

""" Threads """


# Controller

class WorkerThread(QThread):
    processing_complete = pyqtSignal(str)

    def __init__(self, function, param, text_when_done):
        super().__init__()
        self.function = function
        self.param = param
        self.text_when_done = text_when_done


    def run(self):
        # Perform audio processing and predict the genre
        if self.param:
            self.function(self.param)
        else:
            self.function()
        self.processing_complete.emit(self.text_when_done)


class Controller:
    def __init__(self, model, view):
        self.view = view
        self.model = model

        # threads
        self.recording_thread = WorkerThread(self.model.record_audio, GlobalVariables.TRACK_DURATION, text_when_done="Done Recording")
        self.recording_thread.processing_complete.connect(self.append_text_screen)
        self.play_audio_thread = WorkerThread(model.play_audio, param=None, text_when_done="Done Playing")
        self.play_audio_thread.processing_complete.connect(self.append_text_screen)

        # create buttons
        self.load_button = self.view.create_button("Clear Text", self.clear_text)
        self.view.create_circular_button("Record", self.record)
        self.load_button = self.view.create_button("Load Audio", self.load_audio)
        self.view.create_button("Predict Genre", self.predict_genre)
        self.view.create_button("Play Audio", self.play_audio)

        self.view.create_button("Exit", self.exit)
        self.view.show()

    def play_audio(self):
        if not model.is_audio_loaded():
            self.append_text_screen("No audio is loaded")
            return

        self.append_text_screen("Playing..")
        self.view.force_gui_update()
        track_duration_msec = GlobalVariables.TRACK_DURATION * 10

        self.start_bar(track_duration_msec)
        self.play_audio_thread.start()
        # self.play_audio_thread.wait()
        # model.play_audio()

    def append_text_screen(self, text):
        self.view.append_text_output(text)

    def clear_text(self):
        self.view.clear_text_output()

    def start_bar(self, time_ms):
        self.view.progress_bar.setValue(0)
        self.view.recording_timer.start(time_ms)

    def record(self):
        track_duration_msec = GlobalVariables.TRACK_DURATION * 10

        self.append_text_screen("Recording")
        self.view.force_gui_update()
        self.start_bar(track_duration_msec)

        self.recording_thread.start()


    def load_audio(self):
        self.view.append_text_output("Please select a file to upload.")
        filename, _ = QFileDialog.getOpenFileName(self.view, "Open Audio File", "", "Audio Files (*.wav *.mp3)")
        if not filename:
            return
        self.view.append_text_output("Loading audio...")

        self.view.force_gui_update()

        self.model.load_audio(filename=filename)
        self.view.append_text_output("Audio loaded successfully.")

    def predict_genre(self):
        if not self.model.is_audio_loaded():
            self.view.append_text_output("No audio is loaded")
            return

        self.view.append_text_output("Processing...")
        self.view.force_gui_update()

        self.view.append_text_output(self.model.predict())

    def exit(self):
        self.view.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    view = View()
    model = Model()

    controller = Controller(model, view)
    sys.exit(app.exec_())
