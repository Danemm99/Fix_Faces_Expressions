import cv2
import numpy as np
import os
import mediapipe as mp


# Paths to Haar-cascade files for face and smile detection
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
SMILE_CASCADE_PATH = "haarcascade_smile.xml"

# Indices for the lip contour from Mediapipe Face Mesh, used to form the mask
LIP_CONTOUR_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185
]


def create_mouth_mask_for_non_smiling_faces(image_path):
    """
    Algorithm:
      1. Detect faces using Haar-cascade.
      2. Within each face region, detect smiles.
      3. If a smile is NOT detected, run Mediapipe Face Mesh to obtain key landmarks.
      4. From the obtained landmarks, determine the lip contour and compute its bounding box.
      5. Expand the bounding box by a percentage and compute the coordinates of an ellipse
         that is filled into the mask.

    If no faces are found or all faces are smiling, an empty mask is returned.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return mask

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    for (x, y, w, h) in faces:
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = img[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=22)
        if len(smiles) == 0:
            face_roi_rgb = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(face_roi_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                roi_h, roi_w, _ = face_roi_color.shape

                all_landmarks_pixels = []
                for lm in landmarks:
                    lx = int(lm.x * roi_w)
                    ly = int(lm.y * roi_h)
                    all_landmarks_pixels.append([lx + x, ly + y])

                lip_points = []
                for idx in LIP_CONTOUR_INDICES:
                    if idx < len(all_landmarks_pixels):
                        lip_points.append(all_landmarks_pixels[idx])
                if lip_points:
                    lip_points_np = np.array(lip_points, dtype=np.int32)
                    x_min = int(np.min(lip_points_np[:, 0]))
                    y_min = int(np.min(lip_points_np[:, 1]))
                    x_max = int(np.max(lip_points_np[:, 0]))
                    y_max = int(np.max(lip_points_np[:, 1]))

                    width = x_max - x_min
                    height = y_max - y_min
                    margin_x = int(width * 0.5)
                    margin_y = int(height * 0.5)

                    new_x_min = max(0, x_min - margin_x)
                    new_y_min = max(0, y_min - margin_y)
                    new_x_max = min(img.shape[1] - 1, x_max + margin_x)
                    new_y_max = min(img.shape[0] - 1, y_max + margin_y)

                    center = ((new_x_min + new_x_max) // 2, (new_y_min + new_y_max) // 2)
                    axes = ((new_x_max - new_x_min) // 2, (new_y_max - new_y_min) // 2)
                    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=255, thickness=-1)
            else:
                print("Failed to obtain facial landmarks for a non-smiling face.")
    face_mesh.close()
    return mask