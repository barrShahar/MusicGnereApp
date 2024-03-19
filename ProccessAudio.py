import librosa
import numpy as np
import pandas as pd
import GlobalVariables
import sounddevice as sd


def play_audio(audio):
    sd.play(audio, samplerate=GlobalVariables.SAMPLING_RATE)
    sd.wait()


def record_audio(record_time_sec=GlobalVariables.TRACK_DURATION, sampling_rate=GlobalVariables.SAMPLING_RATE):
    """
    Function to record audio using the sounddevice library.

    :return: Recorded audio data as a numpy array and its sampling rate.
    """
    # Query available input devices
    devices = sd.query_devices()

    # Find the first input device
    input_device_id = None
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_device_id = i
            break

    if input_device_id is None:
        raise ValueError("No input devices found.")

    # Record audio for a specific duration at a given sample rate and number of channels
    audio = sd.rec(int(sampling_rate * record_time_sec),
                   samplerate=GlobalVariables.SAMPLING_RATE, channels=device['max_input_channels'],
                   dtype='float32', device=input_device_id)

    # Wait for the recording to finish
    sd.wait()

    # Normalize audio data to the range [-1.0, 1.0]
    audio /= np.max(np.abs(audio))

    # Get the sampling rate
    sampling_rate = GlobalVariables.SAMPLING_RATE

    # Return the recorded audio data and its sampling rate
    return audio, sampling_rate


def extract_middle(y, sr=GlobalVariables.SAMPLING_RATE, duration=GlobalVariables.TRACK_DURATION):
    """
    Extracts the middle segment of an audio signal.

    :param y: Audio signal.
    :param sr: Sampling rate (default value is taken from GlobalVariables).
    :param duration: Desired duration of the segment (default value is taken from GlobalVariables).
    :return: Middle segment of the audio signal.
    """
    total_samples = len(y)  # Total number of samples in the audio signal

    # Calculate the number of samples for the desired duration
    samples_for_duration = duration * sr

    # Ensure that the file is long enough to extract a segment of the desired duration
    if samples_for_duration >= total_samples:
        raise ValueError("The audio duration is shorter than the desired segment duration!")

    # Calculate the starting and ending sample indices for the middle segment
    start = (total_samples - samples_for_duration) // 2
    end = start + samples_for_duration

    # Extract the middle segment of the audio signal
    return y[start:end]


# def trim_and_play(file_name, trim_seconds=GlobalVariables.DURATION, sr=GlobalVariables.SAMPLING_RATE):
#     """
#
#     :param file_name:
#     :param trim_seconds:
#     :param sr:
#     :return:
#     """
#     # Load the audio file
#     y, sr = librosa.load(file_name, sr=sr)
#
#     # extract the middle part
#     y_trimmed = extract_middle(y, duration=trim_seconds)
#
#     # # Play the trimmed audio
#     # sd.play(y_trimmed, samplerate=sr)
#     # sd.wait()
#
#     return y_trimmed
#
#
# def trim(track, trim_seconds=GlobalVariables.DURATION):
#     """
#     Trims the given audio track by extracting the middle segment.
#
#     :param track: Audio track.
#     :param trim_seconds: Duration to trim the audio to (default value is taken from GlobalVariables).
#     :return: Trimmed audio track.
#     """
#
#     # Extract the middle part of the audio track
#     trimmed_track = extract_middle(track, duration=trim_seconds)
#
#     # # Play the trimmed audio (Optional: This part is commented out)
#     # sd.play(trimmed_track, samplerate=sr)
#     # sd.wait()
#
#     return trimmed_track


def extract_features_from_audio(y, song_file_name="Unknown", sr=GlobalVariables.SAMPLING_RATE):
    # Compute features
    print(y.shape)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Organize features into a dictionary
    features = {
        "filename": song_file_name,
        "chroma_stft_mean": np.mean(chroma_stft),
        "chroma_stft_var": np.var(chroma_stft),
        "rms_mean": np.mean(rms),
        "rms_var": np.var(rms),
        "spectral_centroid_mean": np.mean(spectral_centroids),
        "spectral_centroid_var": np.var(spectral_centroids),
        "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
        "spectral_bandwidth_var": np.var(spectral_bandwidth),
        "rolloff_mean": np.mean(rolloff),
        "rolloff_var": np.var(rolloff),
        "zero_crossing_rate_mean": np.mean(zero_crossing_rate),
        "zero_crossing_rate_var": np.var(zero_crossing_rate),
        "harmony_mean": np.mean(harmony),
        "harmony_var": np.var(harmony),
        "perceptr_mean": np.mean(perceptr),
        "perceptr_var": np.var(perceptr),
        "tempo": tempo,
    }

    for i in range(1, 21):
        features[f"mfcc{i}_mean"] = np.mean(mfcc[i - 1])
        features[f"mfcc{i}_var"] = np.var(mfcc[i - 1])

    return pd.DataFrame([features])
