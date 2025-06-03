import cv2
import numpy as np
import os
import mediapipe as mp

# EAR indices for the left and right eyes (from Mediapipe Face Mesh)
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]

# Eye contour indices for creating the eye mask
LEFT_EYE_CONTOUR_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_CONTOUR_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362]

# Threshold for EAR (Eye Aspect Ratio)
EYE_CLOSED_THRESHOLD = 0.22

# OpenCV DNN Face Detector constants
PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_CONFIDENCE_THRESHOLD = 0.10


def calculate_ear(eye_landmarks_pixels):
    try:
        p1 = np.array(eye_landmarks_pixels[0])
        p2 = np.array(eye_landmarks_pixels[1])
        p3 = np.array(eye_landmarks_pixels[2])
        p4 = np.array(eye_landmarks_pixels[3])
        p5 = np.array(eye_landmarks_pixels[4])
        p6 = np.array(eye_landmarks_pixels[5])

        dist_p2_p6 = np.linalg.norm(p2 - p6)
        dist_p3_p5 = np.linalg.norm(p3 - p5)
        dist_p1_p4 = np.linalg.norm(p1 - p4)

        if dist_p1_p4 > 1e-6:
            ear = (dist_p2_p6 + dist_p3_p5) / (2.0 * dist_p1_p4)
        else:
            ear = 1.0
        return ear
    except IndexError:
        print("Error: Incorrect number of points passed to calculate_ear.")
        return 1.0
    except Exception as e:
        print(f"Unknown error in calculate_ear: {e}")
        return 1.0


def create_eye_mask_with_dnn(image_path, prototxt_path, model_path):
    """
    Uses OpenCV DNN to detect faces and then Mediapipe Face Mesh to locate eye landmarks.
    Computes the Eye Aspect Ratio (EAR) and creates a binary mask for closed eyes.
    """
    # Step 1: Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from path: {image_path}")
        return None

    (h, w) = img.shape[:2]

    # Step 2: Load the OpenCV DNN face detector model
    try:
        print("Loading DNN face detection model...")
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("DNN model loaded successfully.")
    except cv2.error as e:
        print("Error: Unable to load DNN model. Check the paths:")
        print(f"Prototxt: '{prototxt_path}'")
        print(f"Model: '{model_path}'")
        print(f"OpenCV error: {e}")
        return None

    # Step 3: Detect faces using DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Step 4: Initialize Mediapipe Face Mesh for landmark detection
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    # Create a blank mask (black image)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Convert original image to RGB once
    img_rgb_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_face_count = 0

    # Step 5: Process each DNN detection
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > DNN_CONFIDENCE_THRESHOLD:
            detected_face_count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Add padding to the ROI
            padding = 20
            roi_startX = max(0, startX - padding)
            roi_startY = max(0, startY - padding)
            roi_endX = min(w, endX + padding)
            roi_endY = min(h, endY + padding)

            if roi_endX <= roi_startX or roi_endY <= roi_startY:
                continue

            face_roi_rgb = img_rgb_original[roi_startY:roi_endY, roi_startX:roi_endX]
            if face_roi_rgb.size == 0:
                continue

            roi_results = face_mesh.process(face_roi_rgb)
            if not roi_results.multi_face_landmarks:
                continue

            # Expect only one face per ROI
            face_landmarks = roi_results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            roi_h, roi_w, _ = face_roi_rgb.shape

            # Process left eye
            try:
                left_eye_ear_points_pixels = []
                for idx in LEFT_EYE_EAR_INDICES:
                    lm = landmarks[idx]
                    orig_x = int(lm.x * roi_w + roi_startX)
                    orig_y = int(lm.y * roi_h + roi_startY)
                    left_eye_ear_points_pixels.append([orig_x, orig_y])
                left_ear = calculate_ear(left_eye_ear_points_pixels)
                if left_ear < EYE_CLOSED_THRESHOLD:
                    left_eye_contour_points = []
                    for idx in LEFT_EYE_CONTOUR_INDICES:
                        lm = landmarks[idx]
                        orig_x = int(lm.x * roi_w + roi_startX)
                        orig_y = int(lm.y * roi_h + roi_startY)
                        left_eye_contour_points.append([orig_x, orig_y])
                    left_eye_contour_points = np.array(left_eye_contour_points, dtype=np.int32)
                    cv2.fillConvexPoly(mask, left_eye_contour_points, 255)
            except IndexError:
                print(f"  Index error processing left eye in ROI (index may exceed {len(landmarks)}).")
            except Exception as e:
                print(f"  Unknown error processing left eye in ROI: {e}")

            # Process right eye
            try:
                right_eye_ear_points_pixels = []
                for idx in RIGHT_EYE_EAR_INDICES:
                    lm = landmarks[idx]
                    orig_x = int(lm.x * roi_w + roi_startX)
                    orig_y = int(lm.y * roi_h + roi_startY)
                    right_eye_ear_points_pixels.append([orig_x, orig_y])
                right_ear = calculate_ear(right_eye_ear_points_pixels)
                if right_ear < EYE_CLOSED_THRESHOLD:
                    right_eye_contour_points = []
                    for idx in RIGHT_EYE_CONTOUR_INDICES:
                        lm = landmarks[idx]
                        orig_x = int(lm.x * roi_w + roi_startX)
                        orig_y = int(lm.y * roi_h + roi_startY)
                        right_eye_contour_points.append([orig_x, orig_y])
                    right_eye_contour_points = np.array(right_eye_contour_points, dtype=np.int32)
                    cv2.fillConvexPoly(mask, right_eye_contour_points, 255)
            except IndexError:
                print(f"  Index error processing right eye in ROI (index may exceed {len(landmarks)}).")
            except Exception as e:
                print(f"  Unknown error processing right eye in ROI: {e}")

    face_mesh.close()
    return mask
