import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import moviepy.editor as mp
import pygame
import time
import threading




interpreter = tf.lite.Interpreter(model_path='models/lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# Input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def predict(frame):
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.float32)
        
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)



video_path = 'songs/copines.mp4'



# Capture all video frames
preds = []

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if ret == False:
        break
    
    frame = cv2.flip(frame,1)
    
    keypoints_with_scores = predict(frame)
    
    preds.append(keypoints_with_scores[0][0])
    # Rendering 
    # draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    # draw_keypoints(frame, keypoints_with_scores, 0.4)

cap.release()
#cv2.destroyAllWindows()

print(len(preds))



clip = mp.VideoFileClip(video_path)
clip = clip.resize((400, 640))
clip = mp.vfx.mirror_x(clip)

#Start the stopwatch
start_time = time.time()

webcam_cap = cv2.VideoCapture(0, cv2.CAP_V4L)
fps = webcam_cap.get(cv2.CAP_PROP_FPS)


def play_clip():
    #time.sleep(0)
    clip.preview(fps)
    pygame.quit()




webcam_preds = []

clip_thread = threading.Thread(target=play_clip, args=())

time.sleep(1)
clip_thread.start()

print(webcam_cap.get(cv2.CAP_PROP_FPS))

while webcam_cap.isOpened():
    time_elapsed = time.time() - start_time

    if time_elapsed > clip.duration + 1.0:
        break

    # if len(webcam_preds) >= len(preds):
    #     break
    
    ret, frame = webcam_cap.read()

    frame = cv2.flip(frame,1)
    
    keypoints_with_scores = predict(frame)
    
    webcam_preds.append(keypoints_with_scores)

    # Rendering 
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    cv2.imshow('MoveNet Lightning', cv2.resize(frame, (480, 640)))
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()




print(len(preds))
print(len(webcam_preds))
