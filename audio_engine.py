import sounddevice as sd
import numpy as np
import threading

SAMPLE_RATE = 44100

audio_frequency = 440.0
audio_volume = 0.0
audio_pinch = False

phase = 0.0
lock = threading.Lock()

def audio_callback(outdata, frames, time, status):
    global phase

    t = np.arange(frames)

    with lock:
        if audio_pinch:
            omega = 2 * np.pi * audio_frequency / SAMPLE_RATE
            samples = audio_volume * np.sin(phase + omega * t)
            phase += omega * frames
        else:
            samples = np.zeros(frames)

    outdata[:] = samples.reshape(-1, 1)

def start_audio():
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=512
    )
    stream.start()
    return stream
