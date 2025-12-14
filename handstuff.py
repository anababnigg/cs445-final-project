import cv2
import mediapipe as mp
import math 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGERTIPS = [4, 8, 12, 16, 20]  

ALPHA = 0.2 

prev_frequency = None
prev_volume = None

from audio_engine import (
    start_audio,
    lock,
    audio_frequency,
    audio_volume,
    audio_pinch
)


def map_controls(x, y, pinch_distance):
    """
    x, y: floats in [0, 1]
    pinch_distance: float (normalized hand landmark distance)
    """

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    frequency = 200 + 800 * x       
    volume = 0.8 * (1.0 - y)            
    pinch = pinch_distance < 0.05    

    return frequency, volume, pinch

def smooth_value(current, previous, alpha):
    """
    Exponential moving average:
    smoothed = alpha * current + (1 - alpha) * previous

    Smoothing reduces jitter because hand tracking produces
    small frame-to-frame noise. EMA dampens rapid fluctuations
    while preserving slower, intentional motion.
    """
    if previous is None:
        return current  
    return alpha * current + (1 - alpha) * previous

def extract_hand_features(hand_landmarks):
    """
    Extracts index fingertip coordinates and computes pinch distance between index and thumb tips.
    """
    index_tip = hand_landmarks.landmark[8]
    index_x, index_y = index_tip.x, index_tip.y

    thumb_tip = hand_landmarks.landmark[4]
    thumb_x, thumb_y = thumb_tip.x, thumb_tip.y

    pinch_distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

    print(f"pinch_distance = {pinch_distance:.4f}")

    return {
        "x": index_x,
        "y": index_y,
        "pinch": pinch_distance
    }

def main():
    cap = cv2.VideoCapture(0)
    stream = start_audio()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:

                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )

                    features = extract_hand_features(hand_landmarks)

                    global prev_frequency, prev_volume

                    frequency, volume, pinch = map_controls(
                        features["x"],
                        features["y"],
                        features["pinch"]
                    )

                    frequency = smooth_value(frequency, prev_frequency, ALPHA)
                    volume = smooth_value(volume, prev_volume, ALPHA)

                    prev_frequency = frequency
                    prev_volume = volume

                    print(f"Smoothed frequency: {frequency:.1f}, volume: {volume:.2f}")

                    from audio_engine import lock
                    import audio_engine
                    
                    with lock:
                        audio_engine.audio_frequency = frequency
                        audio_engine.audio_volume = volume
                        audio_engine.audio_pinch = pinch

                    color = (0, 0, 255) if pinch else (0, 255, 0)
                    label = "PINCH" if pinch else "OPEN"

                    cv2.putText(
                        frame,
                        label,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3
                    )

                    print(features)

                    h, w, c = frame.shape
                    for tip_id in FINGERTIPS:
                        lm = hand_landmarks.landmark[tip_id]
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                        print(f"Fingertip {tip_id}: {cx}, {cy}")

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    stream.stop()
    stream.close()



if __name__ == "__main__":
    main()
