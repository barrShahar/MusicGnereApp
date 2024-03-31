from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QProgressBar, QTextEdit, QApplication


class CircularButton(QPushButton):
    def __init__(self, text="", parent=None):
        super(CircularButton, self).__init__(text, parent)
        self.setMinimumSize(64, 64)
        self.setMaximumSize(64, 64)
        self.setStyleSheet("border-radius: 32px; background-color: red;")
        self.setFont(QFont("Arial", 15, QFont.Bold))  # Set font weight to bold


class View(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Genre Prediction App")
        self.setGeometry(150, 150, 600, 450)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.init_ui()

        # Timer for updating progress
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_progress)

        # Audio and trained model
        # self.loaded_audio = None
        # self.music_classifier = MusicClassifierGenerator(set_default_classifier=True)

        # Thread for recording audio
        # self.recording_thread = RecordThread()
        # self.recording_thread.finished.connect(self.on_recording_finished)

    def init_ui(self):
        # Initialize UI elements
        # self.create_button("Load Audio File", self.load_audio)
        # self.create_button("Predict genre", self.predict_genre)
        # self.create_button("Play Audio", self.play_audio)
        # self.record_button = CircularButton("Record", self)
        # self.record_button.clicked.connect(self.start_recording)

        # self.layout.addWidget(self.record_button, alignment=Qt.AlignCenter)
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.text_output = QTextEdit()
        self.layout.addWidget(self.text_output)
        # self.create_button("Clear Text", self.clear_text)
        # self.create_button("Exit", self.close)

    def create_circular_button(self, text, function):
        circ_button = CircularButton(text, self)
        circ_button.clicked.connect(function)
        self.layout.addWidget(circ_button, alignment=Qt.AlignCenter)
        return circ_button

    def create_button(self, text, function):
        # Create buttons and connect them to functions
        button = QPushButton(text)
        button.clicked.connect(function)
        self.layout.addWidget(button)
        return button

    def append_text_output(self, text):
        self.text_output.append(text)

    def clear_text_output(self):
        self.text_output.clear()

    def force_gui_update(self):
        self.repaint()
        QApplication.processEvents()

    def progress_the_bar(self, time_millsec: int):
        self.progress_bar.setValue(0)
        self.recording_timer.start(time_millsec)  # Start the timer to update progress bar every second

    def update_progress(self):
        # Update progress bar value here
        current_value = self.progress_bar.value()
        max_value = self.progress_bar.maximum()
        if current_value < max_value:
            self.progress_bar.setValue(current_value + 1)
        else:
            self.recording_timer.stop()  # Stop the progress timer