import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        enable_segmentation=True,
        model_complexity=2,
        refine_face_landmarks=True,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image_hight, image_width, _ = image.shape
        if results.pose_landmarks:
            # RIGHT Side
            r_shoulder_x2 = results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_hight
            r_shoulder_y2 = results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_hight
            r_wrist_x1 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_hight
            r_wrist_y1 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_hight
            # LEFT Side
            l_shoulder_x2 = results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_hight
            l_shoulder_y2 = results.pose_landmarks.landmark[
                mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_hight
            l_wrist_x1 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_hight
            l_wrist_y1 = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_hight

            # Derived from the Pythagorean Theorem, I can use this distance formula to find the distance between two points in the plane.
            # In this case the shoulder would be x2y2 and the wrist would be x1y1
            # distance = SQRT((x2 - x1)^2 + (y2 - y1)^2)

            # lets calculate the distance between the right shoulder and right wrist angle
            right_arm_dist = math.sqrt(math.pow((r_shoulder_x2 - r_wrist_x1), 2) +
                                       math.pow((r_shoulder_y2 - r_wrist_y1), 2))
            # lets calcualte the distance between the left shoulder and left wrist angle
            left_arm_dist = math.sqrt(math.pow((l_shoulder_x2 - l_wrist_x1), 2) +
                                      math.pow((l_shoulder_y2 - l_wrist_y1), 2))

            print("right arm radial distance : ", right_arm_dist)
            print("left arm radial distance  : ", left_arm_dist)

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        cv2.imshow('Pose Estimation with Measurements',
                   cv2.flip(annotated_image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
