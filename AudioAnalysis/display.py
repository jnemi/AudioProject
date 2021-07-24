import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load(filePath, sampleRate):
    signal, sr = librosa.load(filePath, sr=sampleRate)
    return signal, sr


def waveform(signal, sr):
    librosa.display.waveplot(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def FFT(signal, sr, reduced=False):
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft)
    frequency = np.linspace(0, sr, len(magnitude))
    if reduced:
        frequency = frequency[:int(len(frequency)/2)]
        magnitude = magnitude[:int(len(magnitude)/2)]
    plt.plot(frequency, magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()


def spectrogram(signal, sr, n_fft=2048, hop_length=512, logarithmic=False):
    # STFT -> spectrogram
    n_fft = 2048  # number of samples
    hop_length = 512  # number of samples

    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

    spectrogram = np.abs(stft)

    if logarithmic:
        spectrogram = librosa.amplitude_to_db(spectrogram)

    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.show()


def MFCCs(signal, sr, n_fft=2048, hop_length=512, n_mfcc=13):
    # MFCCs
    MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCCs")
    plt.colorbar()
    plt.show()
