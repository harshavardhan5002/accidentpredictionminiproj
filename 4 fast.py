import cv2
import numpy as np
import imutils
import math
import pytesseract
from imutils.video import FPS

# -------------------------------------------------------
# TESSERACT CONFIG
# -------------------------------------------------------
# Update this path to your Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------
use_gpu = True
live_video = False
confidence_level = 0.3

# Collision threshold (in meters) for naive collision alert
COLLISION_THRESHOLD = 10.0

# Reference values for naive distance calculation
REF_DISTANCE = 10.0   # meters (arbitrary reference)
REF_WIDTH_PX = 100.0  # pixels (arbitrary reference width)

# Frame skip to reduce processing frequency
FRAME_SKIP = 15  # Process every 5th frame
frame_count = 0

# -------------------------------------------------------
# DISTANCE ESTIMATION (NAIVE)
# -------------------------------------------------------
def estimate_distance(box_width_px):
    """distance ~ (REF_DISTANCE * REF_WIDTH_PX) / box_width_px"""
    if box_width_px <= 0:
        return None
    return (REF_DISTANCE * REF_WIDTH_PX) / box_width_px

# -------------------------------------------------------
# LICENSE PLATE DETECTION & OCR (NAIVE)
# -------------------------------------------------------
def detect_and_read_plate(vehicle_roi):
    """
    1) Convert ROI to gray, bilateral filter, Canny edges
    2) Find 4-corner contour as potential plate
    3) OCR via Tesseract
    Returns text if found, else None
    """
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduce noise, preserve edges
    edged = cv2.Canny(gray, 170, 200)

    cnts, _ = cv2.findContours(edged.copy(),
                               cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate_text = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # 4 corners => possible plate
            x, y, w, h = cv2.boundingRect(approx)
            plate_roi = gray[y:y+h, x:x+w]
            # OCR
            text = pytesseract.image_to_string(plate_roi, config='--psm 8 --oem 3')
            if text.strip():
                plate_text = text.strip()
                break

    return plate_text

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    fps = FPS().start()
    frame_count = 0  # Initialize frame_count inside main()
    
    CLASSES = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"
    ]

    LIGHT_GREEN = (144, 238, 144)
    RED = (0, 0, 255)

    print("[INFO] Loading MobileNet SSD model...")
    net = cv2.dnn.readNetFromCaffe(
        'ssd_files/MobileNetSSD_deploy.prototxt',
        'ssd_files/MobileNetSSD_deploy.caffemodel'
    )

    # GPU?
    if use_gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    print("[INFO] Accessing video stream...")
    if live_video:
        vs = cv2.VideoCapture(0)
    else:
        vs = cv2.VideoCapture('test1.mp4')  # Update with your file path

    recognized_plates = []  # List to store recognized plates

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame_count += 1  # Increment frame_count
        if frame_count % FRAME_SKIP != 0:
            # Skip frames to reduce processing load
            continue

        frame = imutils.resize(frame, width=800)  # Reduced frame resolution
        (h, w) = frame.shape[:2]

        # ------------ 1) Object Detection -------------
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (224, 224), 127.5)  # Reduced blob size
        net.setInput(blob)
        detections = net.forward()

        collision_risk = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_level:
                continue

            idx = int(detections[0, 0, i, 1])
            label_name = CLASSES[idx]

            if label_name not in ["car", "bus", "truck", "motorbike"]:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX   = min(w, endX)
            endY   = min(h, endY)

            box_color = LIGHT_GREEN
            label_text = f"{label_name}: {confidence*100:.2f}%"

            # Distance check
            box_width = endX - startX
            distance = estimate_distance(box_width)
            if distance is not None:
                label_text += f" | Dist: {distance:.2f}m"

                if distance < COLLISION_THRESHOLD:
                    collision_risk = True
                    box_color = RED

                    # 2) If under 10m, attempt license plate detection
                    vehicle_roi = frame[startY:endY, startX:endX].copy()
                    plate_text = detect_and_read_plate(vehicle_roi)
                    if plate_text:
                        label_text += f" | Plate: {plate_text}"
                        print(f"[INFO] Detected Plate: {plate_text}")
                        recognized_plates.append(plate_text)

            # Draw bounding box & label
            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
            label_y = startY - 15 if (startY - 15) > 15 else startY + 15
            cv2.putText(frame, label_text, (startX, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

        # If any collision risk => top-center big text
        if collision_risk:
            warning_text = "COLLISION WARNING"
            font_scale = 1.5
            thickness = 3
            text_size, _ = cv2.getTextSize(warning_text,
                                           cv2.FONT_HERSHEY_DUPLEX,
                                           font_scale, thickness)
            text_width, text_height = text_size
            x_coord = int((w - text_width) / 2)
            y_coord = 60
            cv2.putText(frame, warning_text, (x_coord, y_coord),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale, RED, thickness)

        cv2.imshow('Object, Lane, & Plate Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

        fps.update()

    fps.stop()
    vs.release()
    cv2.destroyAllWindows()

    print("[INFO] Elapsed time: {:.2f} seconds".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    if recognized_plates:
        print("\n[INFO] All recognized plates this session:")
        for i, plate in enumerate(recognized_plates, start=1):
            print(f"  #{i}: {plate}")

if __name__ == "__main__":
    main()


