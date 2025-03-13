import cv2
import time
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open video device")
    exit(1)
else:
    print("Video device opened successfully")

time.sleep(2)  # Add a delay to ensure the camera has time to initialize

cv2.namedWindow("img", cv2.WINDOW_NORMAL)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Skipping frame")
        continue

    print("Frame captured successfully")
    
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img.flags.writeable = True  # Re-enable writeability for drawing on the image

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

    cv2.imshow("img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Exited successfully")






#-