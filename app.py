import sys

from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QFileDialog, \
    QProgressBar
import librosa
import numpy as np
import pandas as pd
import ProccessAudio
import GlobalVariables
from MusicClassifierGenerator import MusicClassifierGenerator


class CircularButtonThread(QPushButton):
    def __init__(self, text="", parent=None):
        super(CircularButtonThread, self).__init__(text, parent)
        self.setMinimumSize(64, 64)
        self.setMaximumSize(64, 64)
        self.setStyleSheet("border-radius: 32px; background-color: red;")
        self.setFont(QFont("Arial", 15, QFont.Bold))  # Set font weight to bold


# class RecordingThread(QThread):
#     def __init__(self, loaded_audio, text_output, parent=None):
#         super().__init__(parent)
#         self.text_output = text_output
#         self.loaded_audio = loaded_audio
#
#     def run(self):
#         self.text_output.append("Recording...")
#         audio = ProccessAudio.record_audio()
#         self.loaded_audio = audio
#         self.text_output.append("Done recording")


class BarProgressThread(QThread):
    def __init__(self, recording_timer):
        super().__init__()
        self.recording_timer = recording_timer

    def run(self):
        pass
        # self.progress_bar.setValue(0)  # Reset progress bar
        self.recording_timer.start(300)  # Start timer with interval of 1 second (1000 milliseconds)


class PredictGenreButton(QThread):
    processing_complete = pyqtSignal(str)

    def __init__(self, loaded_audio, music_classifier):
        super().__init__()
        self.loaded_audio = loaded_audio
        self.music_classifier = music_classifier

    def run(self):
        # Perform audio processing
        features = ProccessAudio.extract_features_from_audio(self.loaded_audio)
        y_pred = self.music_classifier.predict_given_all_features(features)
        result_text = f"Predicted {y_pred}"
        self.processing_complete.emit(result_text)


class AudioProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Processing App")
        self.setGeometry(150, 150, 600, 450)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.button_load = QPushButton("Load Audio File")
        self.button_load.clicked.connect(self.load_audio)
        self.layout.addWidget(self.button_load)

        self.button_process = QPushButton("Predict genre")
        self.button_process.clicked.connect(self.predict_genre)
        self.layout.addWidget(self.button_process)

        self.button_play = QPushButton("Play Audio")
        self.button_play.clicked.connect(self.play_audio)
        self.layout.addWidget(self.button_play)

        self.record_button = CircularButtonThread("Record")
        self.record_button.clicked.connect(self.start_recording)
        self.layout.addWidget(self.record_button)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.text_output = QTextEdit()
        self.layout.addWidget(self.text_output)

        self.button_clear_text = QPushButton("Clear Text")
        self.button_clear_text.clicked.connect(self.clear_text)
        self.layout.addWidget(self.button_clear_text)

        self.button_exit = QPushButton("Exit")
        self.button_exit.clicked.connect(self.close)
        self.layout.addWidget(self.button_exit)

        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_progress)

        self.loaded_audio = None
        self.music_classifier = MusicClassifierGenerator(set_default_classifier=True)
        # self.music_classifier.set_default_classifier()

    def clear_text(self):
        # Clear text from QTextEdit
        self.text_output.clear()

    def load_audio(self):
        # self.text_output.clear()
        self.text_output.append("Loading..")
        filename, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "Audio Files (*.wav *.mp3)")
        if filename:
            self.loaded_audio, _ = librosa.load(filename, sr=None)
            self.loaded_audio = ProccessAudio.extract_middle(self.loaded_audio)
        # check if the duration of the track is >30sc
        self.text_output.append("Audio is Loaded")

    def display_result(self, result_text):
        # Update the text output with the result
        self.text_output.append(result_text)

    def predict_genre(self):
        # self.text_output.clear()
        if self.loaded_audio is not None:
            self.text_output.append("Processing...")

            # Start processing in a separate thread
            processing_thread = PredictGenreButton(self.loaded_audio, self.music_classifier)
            processing_thread.processing_complete.connect(self.display_result)
            processing_thread.start()
            processing_thread.wait()

        else:
            self.text_output.append("No audio is loaded")

    def start_recording(self):
        self.text_output.append("Recording...")
        audio, _ = ProccessAudio.record_audio()
        self.loaded_audio = audio
        # Save the recorded audio as a temporary WAV file

        self.text_output.append("Done recording")

        # # Start recording asynchronously
        # self.recording_thread = RecordingThread(self.loaded_audio, self.text_output)
        # self.recording_thread.text_output = self.text_output  # Pass reference to text output
        # self.recording_thread.start()
        #
        # # Start the progress bar
        # self.progress_bar.setValue(0)
        # self.recording_timer.start(300)  # Start timer with interval of 300 milliseconds

        # self.text_output.append("recording...")
        #
        # # Start recording logic here
        # self.progress_bar.setValue(0)  # Reset progress bar
        # self.recording_timer.start(300)  # Start timer with interval of 1 second (1000 milliseconds)
        #
        # # bar_thread = BarProgressThread(self.recording_timer)
        # # bar_thread.start()
        # audio = ProccessAudio.record_audio()
        # self.loaded_audio = audio
        # self.text_output.append("Done recording")
        # self.play_audio()

    def play_audio(self):
        if self.loaded_audio is not None:
            ProccessAudio.play_audio(self.loaded_audio)

    def update_progress(self):
        # Update progress bar value here
        current_value = self.progress_bar.value()
        max_value = self.progress_bar.maximum()
        if current_value < max_value:
            self.progress_bar.setValue(current_value + 1)
        else:
            self.stop_recording()

    def stop_recording(self):
        # Stop recording logic here
        self.recording_timer.stop()
        # Any other logic to finalize recording


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioProcessingApp()
    window.show()
    sys.exit(app.exec_())
