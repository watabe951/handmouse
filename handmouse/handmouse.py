import cv2
import mediapipe as mp
import time
import winsound
import numpy as np
import pyaudio
import pyautogui

def main():
    def getNearestValue(list, num):
        """
        概要: リストからある値に最も近い値を返却する関数
        @param list: データ配列
        @param num: 対象値
        @return 対象値に最も近い値
        """

        # リスト要素と対象値の差分を計算し最小値のインデックスを取得
        idx = np.abs(np.asarray(list) - num).argmin()
        return list[idx]

    onnkai = {
        "C": 261.626,
        "D": 293.665,
        "E": 329.628,
        "F": 349.228,
        "G": 391.995,
        "A": 440.000,
        "B": 493.883	
    }
    onnkai_value = [v for (k, v) in onnkai.items()]
    SAMPLE_RATE = 44100

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    def to_numpy(data: mp.framework.formats.landmark_pb2.NormalizedLandmark):
        return np.array([data.x, data.y])
    count = 0

    def get_pos(hand):
        return to_numpy(hand.landmark[4])
    def is_click(hand):

        
        # print(hand.landmark[4], hand.landmark[8])
        # print(np.linalg.norm(to_numpy(hand.landmark[4]) - to_numpy(hand.landmark[8])))
        if np.linalg.norm(to_numpy(hand.landmark[4]) - to_numpy(hand.landmark[8])) < 0.03:
            
            return True
        else:
            return False


    is_clicking = False
    pos = None
    try:

        while True:
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            dist = 0
            # print(results.multi_hand_landmarks)
            if results.multi_hand_landmarks:
                # for handLms in results.multi_hand_landmarks:
                handLms = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(img, handLms)
                # print(get_pos(handLms)[0], ", ", get_pos(handLms)[1])
                dist = np.linalg.norm(to_numpy(handLms.landmark[4]) - to_numpy(handLms.landmark[8]))
                if pos is None or np.linalg.norm(get_pos(handLms) - pos) < 1:
                    # print("new_pos")
                    new_pos = get_pos(handLms)
                    if pos is None:
                        pos = new_pos
                    diff_pos = new_pos - pos
                    # print("diff", int(diff_pos[0] * 2200) , int(diff_pos[1]* 1000) ) 
                    if is_clicking:
                        pyautogui.moveRel(int(-diff_pos[0] * 2 * 2200) , int(diff_pos[1]* 2 * 1000) )
                    # pyautogui.moveTo((1 - new_pos[0]) * 2200, new_pos[1] * 1000 - 200)
                # pyautogui.moveTo((1 - new_pos[0]) * 2200, (1 - new_pos[1]) * 1000 - 200)
                    print(is_clicking)
                    if not is_click(handLms):
                        is_clicking = False
                    elif is_clicking == False and is_click(handLms):
                        is_clicking = True

                    # if is_click(handLms) and is_clicking == False:
                    #     is_clicking = True
                    #     count += 1
                    #     # pyautogui.click()
                    #     # print("click", count)
                    # else:
                    #     is_clicking = False
                    #     # print("not click", count)
                    
                    pos = new_pos
                
            flipHorizontal = cv2.flip(img, 1)
            cv2.putText(flipHorizontal, f'{dist}', (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)

            cv2.imshow("Image", flipHorizontal)
            key = cv2.waitKey(1)
            if key == ord("r"):
                print("Key: \'" + chr(key) + "\' pressed.")

    except KeyboardInterrupt:
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()
