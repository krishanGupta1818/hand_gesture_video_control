import cv2
import mediapipe as mp
import pyautogui
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, handedness='Right'):
    if hand_landmarks is None:
        return 0
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb
    if handedness == 'Right':
        fingers.append(1 if lm[4].x < lm[3].x - 0.02 else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x + 0.02 else 0)

    # Other fingers
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(1 if lm[tip].y < lm[pip].y - 0.02 else 0)

    return sum(fingers)

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

finger_buffer = deque(maxlen=15)
gesture_hold = {'count': None, 'frames': 0, 'last_trigger_time': 0}
current_action = "WAITING..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    detected_fingers = 0
    handedness = 'Right'
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
        detected_fingers = count_fingers(hand_landmarks, handedness)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    finger_buffer.append(detected_fingers)
    smoothed = max(set(finger_buffer), key=finger_buffer.count)

    if gesture_hold['count'] == smoothed:
        gesture_hold['frames'] += 1
    else:
        gesture_hold['count'] = smoothed
        gesture_hold['frames'] = 1

    if gesture_hold['frames'] >= 10 and (time.time() - gesture_hold['last_trigger_time'] > 0.7):
        g = gesture_hold['count']
        gesture_hold['last_trigger_time'] = time.time()

        if g == 5:
            pyautogui.press('k')
            current_action = "PLAY / PAUSE"
        elif g == 0:
            pyautogui.press('k')
            current_action = "PLAY / PAUSE"
        elif g == 2:
            pyautogui.press('l')
            current_action = "FORWARD 10s"
        elif g == 3:
            pyautogui.press('j')
            current_action = "BACKWARD 10s"
        else:
            current_action = "UNKNOWN"

    # Show clear big text on screen
    cv2.putText(frame, f"Gesture: {current_action}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"Fingers: {smoothed}", (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YouTube Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
