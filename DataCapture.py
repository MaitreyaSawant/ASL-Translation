import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import time


mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=True
    image=cv.cvtColor(image,cv.COLOR_RGB2BGR)
    return image,results

def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,  mp_holistic.FACEMESH_CONTOURS,mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),mp_drawing.DrawingSpec(color=(80,256,110), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# cap= cv.VideoCapture(0)
# with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
#     while True:
#         ret,frame=cap.read()
#
#         image,results=mediapipe_detection(frame,holistic)
#         print(results)
#         draw_landmarks(image,results)
#
#         cv.imshow('OpenCV Feed', image)
#
#         if cv.waitKey(10) & 0xFF ==ord('q'):
#              break
#
#
# cap.release()
# cv.destroyAllWindows()

def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face=np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])

DATA_PATH=os.path.join('MP_Data')
actions=np.array(['hello','thanks','iloveyou'])
no_sequences=30
sequence_length=30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
        except:
            pass

cap= cv.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range (sequence_length):


                ret,frame=cap.read()

                image,results=mediapipe_detection(frame,holistic)
                print(results)
                draw_landmarks(image,results)

                if frame_num==0:
                    cv.putText(image,"STARTING COLLECTION",(120,200),cv.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv.LINE_AA)
                    cv.putText(image,"COLLECTING FRAMES FOR {} VIDEO NUMBER {}".format(action,sequence),(15,12),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)
                    cv.waitKey(2000)
                else:
                    cv.putText(image,"Collecting frames for {} Video Number {}".format(action,sequence),(15,12),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv.LINE_AA)

                keypoints=extract_keypoints(results)
                npy_path=os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                np.save(npy_path,keypoints)

                if not os.path.exists(f"{npy_path}.npy"):
                    np.save(npy_path, keypoints)
                else:
                    print(f"File {npy_path}.npy already exists")


                cv.imshow('OpenCV Feed', image)

                if cv.waitKey(10) & 0xFF ==ord('q'):
                     break


cap.release()
cv.destroyAllWindows()