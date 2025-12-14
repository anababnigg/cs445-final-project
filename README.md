# CS 445 Final Project: Hand-Tracked Musical Controller

Webcam-based hand controller that maps simple gestures to real-time sound. Built with MediaPipe Hands + OpenCV and a Python audio engine.

Features
- Pitch and volume from index fingertip position
- Pinch to gate sound (thumb + index)
- Fist to hold the current pitch (open to resume live control)
- Octave shift via pinches:
  - Thumb + middle → octave up (+1)
  - Thumb + ring → octave down (−1)
- On-screen HUD (top-right): Freq, Vol, Pinch, Hold, Oct, FPS + visual bars

Demo keys
- Press q or ctl+c in terminal to quit the window

---

## Requirements

- Python 3.11 recommended
- Webcam
- Packages (tested versions):
  - mediapipe==0.10.14
  - opencv-python==4.10.0.84
  - numpy==1.26.4
  - sounddevice

Note: some operations use an earlier (1.x) form of numpy. If issues happen, try using an older version of numpy.

---

## Setup

Using Conda
```bash
conda create -n cs445-final python=3.11 -y
conda activate cs445-final
python -m pip install -U pip
pip install mediapipe==0.10.14 opencv-python==4.10.0.84 numpy==1.26.4
pip install sounddevice
```

---
### Use
Note: the camera view is mirrored.

- Gate sound (note on/off): thumb + index pinch
- Control pitch: move index fingertip left <-> right (base range ~200–1000 Hz, plus octave shift)
- Control volume: move index fingertip up <-> down (higher on screen = louder)
- Hold pitch: make a fist to latch current pitch; open your hand to resume live control
- Octave shift:
    - Thumb + middle pinch -> Oct: +1
    - Thumb + ring pinch -> Oct: −1

HUD Elements:
- Text: Freq, Vol, Pinch (with normalized pinch distance), Hold, Oct, FPS
- Bars: vertical VOL bar and horizontal FREQ bar

---
### Acknoledgements
- Mediapipe hands by Google
- OpenCV
- Inspiration: hand-gestural control systems (Imogen Heap’s Mi.Mu gloves)


