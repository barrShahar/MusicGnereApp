import librosa
import numpy as np
from PyQt5.QtWidgets import QFileDialog

import GlobalVariables
import ProccessAudio
from MusicClassifierGenerator import MusicClassifierGenerator


class Model:
    def __init__(self):
        self._music_classifier = MusicClassifierGenerator(train_default_classifier=True)
        self._audio = None

    def load_audio(self, filename):
        # self.text_output.clear()
        self._audio, _ = librosa.load(filename, sr=GlobalVariables.SAMPLING_RATE)
        self._audio = ProccessAudio.extract_middle(self._audio)
        # check if the duration of the track is >30sc

    def predict(self):
        # Predict genre of the loaded audio
        features = ProccessAudio.extract_features_from_audio(self._audio)
        y_pred = self._music_classifier.predict_given_all_features(features)
        result_text = f"Predicted {y_pred}"
        return result_text

    def is_audio_loaded(self):
        if self._audio is None:
            return False
        return True

    def record_audio(self, record_time_sec):
        audio, _ = ProccessAudio.record_audio(record_time_sec, GlobalVariables.SAMPLING_RATE)
        self._audio = np.reshape(audio, (record_time_sec * GlobalVariables.SAMPLING_RATE,))

    def play_audio(self):
        ProccessAudio.play_audio(self._audio)