# import cv2
# from deepface import DeepFace

# # Load face cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start capturing video
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Convert frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Convert grayscale frame to RGB format
#     rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face ROI (Region of Interest)
#         face_roi = rgb_frame[y:y + h, x:x + w]

        
#         # Perform emotion analysis on the face ROI
#         result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

#         # Determine the dominant emotion
#         emotion = result[0]['dominant_emotion']

#         # Draw rectangle around face and label with predicted emotion
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Real-time Emotion Detection', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close all windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import speech_recognition as sr  # For ASR
import pyaudio
import wave
import threading 

# Initialize Mediapipe Pose and Drawing Utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize ASR
# recognizer = sr.Recognizer()
# mic = sr.Microphone()
# Audio recording settings

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
OUTPUT_WAV = "recorded_audio.wav"

# Function to record audio in a separate thread
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    print("Recording audio...")
    while audio_recording:
        data = stream.read(CHUNK)
        frames.append(data)

    print("Audio recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the audio to a file
    with wave.open(OUTPUT_WAV, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Thresholds for posture
POSTURE_THRESHOLD = 0.5
FORWARD_POSTURE_THRESHOLD = -0.2

# Array to store shoulder data
shoulder_data = []

# Start audio recording in a separate thread
audio_recording = True
audio_thread = threading.Thread(target=record_audio)
audio_thread.start()

# Transcription variable
# transcription = "Listening..."

# Initialize pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB for Mediapipe and DeepFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process image and extract pose landmarks
        results = pose.process(rgb_frame)

        # Extract posture landmarks and analyze
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get the x and z coordinates of the left and right shoulders
            left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
            right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
            left_shoulder_z = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            right_shoulder_z = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z

            # Append data to the array
            shoulder_data.append([left_shoulder_x, right_shoulder_x, left_shoulder_z, right_shoulder_z])

            # Determine if the posture is front-facing or side-facing
            shoulder_diff_x = abs(left_shoulder_x - right_shoulder_x)
            posture_orientation = "Front Facing" if shoulder_diff_x < POSTURE_THRESHOLD else "Side Facing"

            # Determine if the posture is forward-leaning
            avg_shoulder_z = (left_shoulder_z + right_shoulder_z) / 2
            posture_depth = "Forward Posture" if avg_shoulder_z < FORWARD_POSTURE_THRESHOLD else "Good Posture"

            # Combine posture status
            posture_status = f"{posture_orientation} - {posture_depth}"

            # Display posture status on the image
            cv2.putText(frame, posture_status,
                        (50, 50),  # Text position
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if posture_depth == "Good Posture" else (0, 0, 255),
                        2, cv2.LINE_AA)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Focus on the main (largest) face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            # Extract the face ROI
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform emotion analysis
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                emotion = "N/A"

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

        # # Display transcription (ASR)
        # cv2.putText(frame, f"Transcription: {transcription}",
        #             (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Emotion and Posture Analysis', frame)

        # # Perform ASR in the background
        # try:
        #     with mic as source:
        #         recognizer.adjust_for_ambient_noise(source)
        #         audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
        #         transcription = recognizer.recognize_google(audio)
        # except sr.WaitTimeoutError:
        #     transcription = "Listening..."
        # except sr.UnknownValueError:
        #     transcription = "Could not understand."
        # except sr.RequestError:
        #     transcription = "ASR service error."

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Stop audio recording and save data
audio_recording = False
audio_thread.join()

# Save shoulder data to a CSV file
shoulder_data = np.array(shoulder_data)
np.savetxt("shoulder_data.csv", shoulder_data, delimiter=",",
           header="left_shoulder_x,right_shoulder_x,left_shoulder_z,right_shoulder_z", comments="")

cap.release()
cv2.destroyAllWindows()
