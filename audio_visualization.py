import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_waveform(file_path):
    y, sr = librosa.load(file_path)

    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path)

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure()
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram")
    plt.show()