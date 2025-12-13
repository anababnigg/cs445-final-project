import cv2
import mediapipe as mp
import math 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGERTIPS = [4, 8, 12, 16, 20]  

def extract_hand_features(hand_landmarks):
    """
    Extracts index fingertip coordinates and computes pinch distance between index and thumb tips.
    """
    index_tip = hand_landmarks.landmark[8]
    index_x, index_y = index_tip.x, index_tip.y

    thumb_tip = hand_landmarks.landmark[4]
    thumb_x, thumb_y = thumb_tip.x, thumb_tip.y

    pinch_distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

    return {
        "x": index_x,
        "y": index_y,
        "pinch": pinch_distance
    }

def main():
    cap = cv2.VideoCapture(0)

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


if __name__ == "__main__":
    main()
