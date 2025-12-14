# -- initialize --
import cv2
import mediapipe as mp
import math 
import time 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGERTIPS = [4, 8, 12, 16, 20]  
ALPHA = 0.2 

prev_frequency = None
prev_volume = None

# -- audio --
from audio_engine import (
    start_audio,
    lock,
    audio_frequency,
    audio_volume,
    audio_pinch
)
# ------------

# -- HUD --
def clamp01(v):
    return max(0.0, min(1.0, v))

def draw_hud(frame, frequency, volume, pinch, x, y, pinch_distance,
             fps=None, anchor='tr', show_big_label=False, hold=False, octave=0):
    h, w, _ = frame.shape
    panel_w, panel_h = 360, 230
    margin = 10
    anchor = anchor.lower()
    if anchor in ('tr', 'top-right'):
        x0, y0 = w - margin - panel_w, margin
    elif anchor in ('br', 'bottom-right'):
        x0, y0 = w - margin - panel_w, h - margin - panel_h
    elif anchor in ('bl', 'bottom-left'):
        x0, y0 = margin, h - margin - panel_h
    else:  # 'tl'
        x0, y0 = margin, margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # normalize
    vol_norm = clamp01(volume / 0.8)               
    freq_norm = clamp01((frequency - 200) / 800)   

    # Text layout
    lh = 24  # line height
    y_text = y0 + 28

    # -> outputs
    # freq + FPS 
    cv2.putText(frame, f"Freq: {frequency:.0f} Hz",
                (x0 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (x0 + panel_w - 90, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    y_text += lh
    # vol line
    cv2.putText(frame, f"Vol: {volume:.2f}",
                (x0 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
    y_text += lh
    # pinch line
    pinch_color = (0, 0, 255) if pinch else (0, 255, 0)
    cv2.putText(frame, f"Pinch: {'ON' if pinch else 'OFF'}   d={pinch_distance:.3f}",
                (x0 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pinch_color, 2)
    y_text += lh
    # "hold" line
    cv2.putText(frame, f"Hold: {'ON' if hold else 'OFF'}",
                (x0 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_text += lh
    # octave 
    cv2.putText(frame, f"Oct: {octave:+d}", 
                (x0 + 150, y_text - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    # bars (placed below text block)
    bar_x, bar_y, bar_w, bar_h = x0 + 10, y0 + panel_h - 90, 24, 70
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_h = int(bar_h * vol_norm)
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), (0, 255, 0), -1)
    cv2.putText(frame, "VOL", (bar_x - 2, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    fx, fh = x0 + 60, 12
    fy = y0 + panel_h - 26
    fw = panel_w - 80
    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (60, 60, 60), -1)
    cv2.rectangle(frame, (fx, fy), (fx + int(fw * freq_norm), fy + fh), (255, 200, 0), -1)
    cv2.putText(frame, "FREQ", (fx, fy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # highlights index (not sure if this works)
    xi, yi = int(x * w), int(y * h)
    cv2.circle(frame, (xi, yi), 5, (255, 255, 255), -1)
# ----------

# --- "hold" mechanic ---
OPEN_ON  = 1.05  
OPEN_OFF = 0.95  

def hand_openness_score(hand_landmarks):
    """
    avg distance of fingertips to palm center, normalized by palm width.
    Higher = more open, lower = more closed (fist).
    """
    lm = hand_landmarks.landmark
    pcx = (lm[0].x + lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 5.0
    pcy = (lm[0].y + lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 5.0
    palm_w = max(1e-6, math.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y))
    tips = [8, 12, 16, 20] 
    d = [math.hypot(lm[t].x - pcx, lm[t].y - pcy) / palm_w for t in tips]
    return sum(d) / len(d)
# --------------------------------

# --- octave shift ---
PINCH_ON  = 0.24   # normalized distance thresh for pinch
PINCH_OFF = 0.30   
SHIFT_DEBOUNCE_MS = 300  # min time between shifts

def normalized_pinch(hand_landmarks, tip_id_a, tip_id_b):
    """Normalized distance between two landmarks, using palm width as scale."""
    lm = hand_landmarks.landmark
    ax, ay = lm[tip_id_a].x, lm[tip_id_a].y
    bx, by = lm[tip_id_b].x, lm[tip_id_b].y
    d = math.hypot(ax - bx, ay - by)
    palm_w = max(1e-6, math.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y))
    return d / palm_w
# --------------------------------------------------

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

    # -- HUD  tracking ----
    prev_time = time.time()
    fps = 0.0
    # ----------------------
    open_state = True      
    hold_active = False
    held_freq = None
    # octave states
    octave_shift = 0          
    mid_pinch = False        
    ring_pinch = False        
    last_mid_ms = 0.0
    last_ring_ms = 0.0

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
            # -- HUD ---
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  
            # ----------

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
                    # --- OPEN/FIST detection (for HOLD) ---
                    open_score = hand_openness_score(hand_landmarks)
                    prev_open_state = open_state
                    if open_state:
                        if open_score < OPEN_OFF:
                            open_state = False  
                    else:
                        if open_score > OPEN_ON:
                            open_state = True   
                    # --------------------------------------

                    # --- octave shift detection ---
                    t_ms = time.time() * 1000.0
                    mid_norm = normalized_pinch(hand_landmarks, 4, 12)   
                    ring_norm = normalized_pinch(hand_landmarks, 4, 16) 
                    # middle pinch = octave up 
                    if not mid_pinch and mid_norm < PINCH_ON and (t_ms - last_mid_ms) > SHIFT_DEBOUNCE_MS:
                        mid_pinch = True
                        last_mid_ms = t_ms
                        octave_shift = min(octave_shift + 1, 2)   
                    elif mid_pinch and mid_norm > PINCH_OFF:
                        mid_pinch = False
                    # rinch pinch = octave down
                    if not ring_pinch and ring_norm < PINCH_ON and (t_ms - last_ring_ms) > SHIFT_DEBOUNCE_MS:
                        ring_pinch = True
                        last_ring_ms = t_ms
                        octave_shift = max(octave_shift - 1, -2)  
                    elif ring_pinch and ring_norm > PINCH_OFF:
                        ring_pinch = False
                    # --------------------------------------------------------------

                    global prev_frequency, prev_volume

                    frequency, volume, pinch = map_controls(
                        features["x"],
                        features["y"],
                        features["pinch"]
                    )

                    # -> octave shift 
                    frequency *= (2.0 ** octave_shift)
                    frequency = max(50.0, min(2000.0, frequency))

                    # -> if closed hand, hold
                    if (prev_open_state and not open_state):
                        hold_active = True
                        held_freq = prev_frequency if prev_frequency is not None else frequency
                    # -> release hold
                    if (not prev_open_state and open_state):
                        hold_active = False
                    # -> apply hold
                    if hold_active and held_freq is not None:
                        frequency = held_freq

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

                    # -- HUD --
                    # draw the HUD
                    draw_hud(
                        frame,
                        frequency=frequency,
                        volume=volume,
                        pinch=pinch,
                        x=features["x"],
                        y=features["y"],
                        pinch_distance=features["pinch"],
                        fps=fps,
                        anchor='tr',
                        show_big_label=False,
                        hold=hold_active,
                        octave=octave_shift
                    )
                    # ---------

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
