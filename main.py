import AudioAnalysis.display as display

file = 'data/diarizationExample.wav'

signal, sr = display.load(file, 2048)
# display.waveform(signal, sr)
# display.spectrogram(signal, sr, n_fft=2048, hop_length=512, logarithmic=False)
# display.FFT(signal, sr, reduced=True)
display.MFCCs(signal, sr, n_fft=2048, hop_length=2048, n_mfcc=13)
