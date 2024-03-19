# Configuration variables:

# Flag to control plotting
g_plot = False
dataset_path = "./csv/data3.csv"

gtzan_genres = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

selected_genre = [
    "classical",
    "pop", "jazz", "rock"
]

# Directory paths for various machine learning models:
# g_knn_folder = r'/Users/shaharbarr/Documents/Workbench/ML music project/musicPythionProjectML/knn/'
# g_SVM_folder = r'/Users/shaharbarr/Documents/Workbench/ML music project/musicPythionProjectML/SVM/'
# g_adaboost_folder = r'/Users/shaharbarr/Documents/Workbench/ML music project/musicPythionProjectML/adaboost/'
# g_perceptron = r'/Users/shaharbarr/Documents/Workbench/ML music project/musicPythionProjectML/perceptron/'
#
# # Feature extraction parameters:
# MUSIC_FOLDER_PATH = r'/Users/shaharbarr/Documents/Workbench/music/genres/blues/'  # Path to the music genre folder
# LABEL = 'rock'            # Genre label for the audio files
SAMPLING_RATE = 22050     # Audio sampling rate (samples/second)
MAX_AUDIO_FILES = 300     # Maximum number of audio files to process
TRACK_DURATION = 30             # Duration for middle segment extraction (in seconds)
