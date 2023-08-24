import cv2
import mediapipe as mp
import time
from pose_buffer import PoseBuffer
import numpy as np

def compare_poses(pose_landmarks1, pose_landmarks2):
    keypoints1 = np.array([(landmark.x, landmark.y) for landmark in pose_landmarks1.landmark])
    keypoints2 = np.array([(landmark.x, landmark.y) for landmark in pose_landmarks2.landmark])
    
    # Normalize keypoints
    keypoints1 /= np.linalg.norm(keypoints1, axis=0)
    keypoints2 /= np.linalg.norm(keypoints2, axis=0)
    
    # Calculate cosine similarity
    similarity_score = np.dot(keypoints1.flatten(), keypoints2.flatten())
    return similarity_score

def main():
    # Initialize Mediapipe Pose model
    mp_pose = mp.solutions.pose
    
    pose = mp_pose.Pose()


    # Open the video file
    video_path = "vid.mp4"  # Replace with the path to your video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Capture the video frame rate
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    

    # Initialize webcam
    webcam = cv2.VideoCapture(0)  # Use the default webcam (change index if you have multiple webcams)

    mp_drawing = mp.solutions.drawing_utils

    video_buffer = PoseBuffer()
    webcam_buffer = PoseBuffer()

    while cap.isOpened() and webcam.isOpened():

        timestamp = time.time()

        # Read frames from video and webcam
        ret, video_frame = cap.read()
        
        #ret_webcam, webcam_frame = webcam.read()

        """ if not ret or not ret_webcam:
            continue """

        # Convert frames to RGB (Mediapipe expects RGB images)
        rgb_video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        #rgb_webcam_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)

        # Perform pose estimation on video frame
        video_results = pose.process(rgb_video_frame)

        # Perform pose estimation on webcam frame
        #webcam_results = pose.process(rgb_webcam_frame)
        
        if video_results.pose_landmarks:
            video_buffer.add_pose(timestamp, video_results.pose_landmarks)
            
            # Draw pose landmarks on video frame
            mp_drawing.draw_landmarks(video_frame, video_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        """ if webcam_results.pose_landmarks:
            webcam_buffer.add_pose(timestamp, webcam_results.pose_landmarks)

            # Draw pose landmarks on webcam frame
            mp_drawing.draw_landmarks(webcam_frame, webcam_results.pose_landmarks, mp_pose.POSE_CONNECTIONS) """
        
        # Compare pose landmarks and calculate similarity score using buffered poses
        similarity_score = 0
        
        """ latest_webcam_timestamp, latest_webcam_pose = webcam_buffer.get_latest_pose()

        # Find matching video pose using timestamps
        matching_video_pose = None
        for timestamp, pose_landmarks in video_buffer.buffer:
            if timestamp == latest_webcam_timestamp:
                matching_video_pose = pose_landmarks
                break """
        
        #if matching_video_pose is not None:
        similarity_score = compare_poses(video_results.pose_landmarks, video_results.pose_landmarks)

        # Display feedback and score on the frames
        cv2.putText(video_frame, f"Similarity: {similarity_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(webcam_frame, f"Similarity: {similarity_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize frames to fill 50% of the screen each
        frame_height, frame_width, _ = video_frame.shape
        video_frame = cv2.resize(video_frame, (600, 500))
        #webcam_frame = cv2.resize(webcam_frame, (600, 500))

        # Combine frames side by side
        #combined_frame = cv2.hconcat([video_frame, webcam_frame])

        # Display combined frame
        cv2.imshow("Dance Comparison and Scoring", video_frame)

        if cv2.waitKey(int(1000 / video_fps)) & 0xFF == ord('q'):
            break
        
        #time.sleep(0.1)

    cap.release()
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()