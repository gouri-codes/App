import librosa
import numpy as np


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)

        # MFCC (main audio features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0])

        # Energy
        energy = np.mean(librosa.feature.rms(y=y))

        features = list(mfcc_mean)
        features.append(pitch)
        features.append(energy)

        return features

    except:
        return None