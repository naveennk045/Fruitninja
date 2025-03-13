import cv2
import time
import random
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

curr_Frame = 0
prev_Frame = 0
delta_time = 0

next_Time_to_Spawn = 0
Speed = [0, 5]
Fruit_Size = 60
Spawn_Rate = 1
Score = 0
Lives = 15
Difficulty_level = 1
game_Over = False

slash = np.array([[]], np.int32)
slash_Color = (255, 255, 255)
slash_length = 19

w = h = 0

Fruits = []

# Load fruit images
fruit_images = [
    cv2.imread(r"/Fruitninja/fruitimage/image1.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image2.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image3.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image4.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image5.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image6.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image7.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image8.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image9.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image11.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image10.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image12.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image9.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image9.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image8.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image8.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image2.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image3.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image5.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image13.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image14.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image13.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image16.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image17.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image18.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image19.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image20.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image21.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image22.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image23.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image24.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image25.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image26.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image27.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image28.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image29.png", cv2.IMREAD_UNCHANGED),
    cv2.imread(r"/Fruitninja/fruitimage/image30.png", cv2.IMREAD_UNCHANGED),

]

# Check if images were loaded successfully and convert to BGRA if needed
for idx, img in enumerate(fruit_images):
    if img is None:
        print(f"Error loading image at index {idx}")
        fruit_images[idx] = np.zeros((Fruit_Size, Fruit_Size, 4), dtype=np.uint8)  # Placeholder for missing image
    elif img.shape[2] == 3:  # If the image doesn't have an alpha channel
        print(f"Converting image at index {idx} to BGRA")
        fruit_images[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

def Spawn_Fruits():
    fruit = {}
    random_x = random.randint(15, 600)
    random_fruit_img = random.choice(fruit_images)
    fruit["Image"] = random_fruit_img
    fruit["Curr_position"] = [random_x, 440]
    fruit["Next_position"] = [0, 0]
    Fruits.append(fruit)

def Fruit_Movement(Fruits, speed):
    global Lives, img

    fruits_to_remove = []

    for fruit in Fruits:
        if (fruit["Curr_position"][1]) < 20 or (fruit["Curr_position"][0]) > 650:
            Lives = Lives - 1
            fruits_to_remove.append(fruit)
            continue

        fruit_img = fruit["Image"]
        fruit_pos = fruit["Curr_position"]
        fruit_size = (Fruit_Size, Fruit_Size)
        fruit_img_resized = cv2.resize(fruit_img, fruit_size)

        # Ensure the fruit is within the image boundaries
        y1, y2 = fruit_pos[1], fruit_pos[1] + Fruit_Size
        x1, x2 = fruit_pos[0], fruit_pos[0] + Fruit_Size

        # Ensure the indices are within the image bounds
        if y2 > img.shape[0]:
            y2 = img.shape[0]
        if x2 > img.shape[1]:
            x2 = img.shape[1]
        if y1 < 0:
            y1 = 0
        if x1 < 0:
            x1 = 0

        fruit_height, fruit_width = y2 - y1, x2 - x1
        fruit_img_cropped = fruit_img_resized[:fruit_height, :fruit_width]

        alpha_s = fruit_img_cropped[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * fruit_img_cropped[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

        fruit["Next_position"][0] = fruit["Curr_position"][0] + speed[0]
        fruit["Next_position"][1] = fruit["Curr_position"][1] - speed[1]

        fruit["Curr_position"] = fruit["Next_position"]

    # Remove fruits after iterating over them
    for fruit in fruits_to_remove:
        Fruits.remove(fruit)

def distance(a, b):
    x1 = a[0]
    y1 = a[1]

    x2 = b[0]
    y2 = b[1]

    d = math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    return int(d)

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    success, img = cap.read()
    if not success:
        print("Skipping frame")
        continue
    h, w, c = img.shape
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for id, lm in enumerate(hand_landmarks.landmark):
                if id == 8:
                    index_pos = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(img, index_pos, 18, slash_Color, -1)
                    slash = np.append(slash, index_pos)

                    while len(slash) >= slash_length:
                        slash = np.delete(slash, len(slash) - slash_length, 0)

                    fruits_to_remove = []

                    for fruit in Fruits:
                        d = distance(index_pos, fruit["Curr_position"])
                        cv2.putText(img, str(d), tuple(fruit["Curr_position"]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, 3)
                        if (d < Fruit_Size):
                            Score = Score + 100
                            slash_Color = (255, 255, 255)
                            fruits_to_remove.append(fruit)

                    # Remove fruits after iterating over them
                    for fruit in fruits_to_remove:
                        Fruits.remove(fruit)

    if Score % 1000 == 0 and Score != 0:
        Difficulty_level = (Score / 1000) + 1
        Difficulty_level = int(Difficulty_level)
        Spawn_Rate = Difficulty_level * 4 / 5
        Speed[0] = Speed[0] * Difficulty_level
        Speed[1] = int(5 * Difficulty_level / 2)

    if (Lives <= 0):
        game_Over = True

    slash = slash.reshape((-1, 1, 2))
    cv2.polylines(img, [slash], False, slash_Color, 15, 0)

    curr_Frame = time.time()
    delta_Time = curr_Frame - prev_Frame
    FPS = int(1 / delta_Time)
    cv2.putText(img, "FPS : " + str(FPS), (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
    cv2.putText(img, "Score: " + str(Score), (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.putText(img, "Level: " + str(Difficulty_level), (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
    cv2.putText(img, "Lives remaining : " + str(Lives), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    prev_Frame = curr_Frame

    if not (game_Over):
        if (time.time() > next_Time_to_Spawn):
            Spawn_Fruits()
            next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)

        Fruit_Movement(Fruits, Speed)
    else:
        cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        Fruits.clear()

    cv2.imshow("img", img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()