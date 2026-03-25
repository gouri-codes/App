import sounddevice as sd
import scipy.io.wavfile as wav


def record_audio(filename, duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("Saved:", filename)