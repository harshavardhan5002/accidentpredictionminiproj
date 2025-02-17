import cv2
import numpy as np
import imutils
import math
import pytesseract
from imutils.video import FPS
import socket

# -------------------------------------------------------
# TESSERACT CONFIG
# -------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------------------------------------------------------
# CONFIGURATIONS
# -------------------------------------------------------
use_gpu = True
live_video = False
confidence_level = 0.3
person_confidence_threshold = 0.2  # Lower threshold for pedestrians

COLLISION_THRESHOLD = 10.0  # Collision threshold in meters

REF_DISTANCE = 10.0  # Reference distance in meters
REF_WIDTH_PX = 100.0  # Reference width in pixels

# Server IP and Port (Raspberry Pi's IP)
RPI_IP = '192.168.29.43'  # Change this to your Raspberry Pi's IP address
RPI_PORT = 65432

# Create a socket connection to send data to RPI
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((RPI_IP, RPI_PORT))


# -------------------------------------------------------
# DISTANCE ESTIMATION
# -------------------------------------------------------
def estimate_distance(box_width_px):
    if box_width_px <= 0:
        return None
    return (REF_DISTANCE * REF_WIDTH_PX) / box_width_px


# -------------------------------------------------------
# LICENSE PLATE DETECTION & OCR
# -------------------------------------------------------
def detect_and_read_plate(vehicle_roi):
    gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate_text = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            plate_roi = gray[y:y + h, x:x + w]
            text = pytesseract.image_to_string(plate_roi, config='--psm 8 --oem 3')
            if text.strip():
                plate_text = text.strip()
                break

    return plate_text


# -------------------------------------------------------
# LANE DETECTION
# -------------------------------------------------------
def process_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    imshape = frame.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] // 2, imshape[0] // 2 + 50),
                          (imshape[1] // 2, imshape[0] // 2 + 50), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = cv2.bitwise_and(edges, cv2.fillPoly(np.zeros_like(edges), vertices, 255))

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=20)
    lines_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)

    avg_angle = 0  # Default value if no lines are detected
    if lines is not None:
        angles = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Avoid division by zero
                angle = math.degrees(math.atan(slope))
                angles.append(angle)

        if angles:
            avg_angle = np.mean(angles)

    result = cv2.addWeighted(frame, 0.8, lines_img, 1, 0)
    return result, avg_angle


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def main():
    fps = FPS().start()

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    LIGHT_GREEN = (144, 238, 144)
    RED = (0, 0, 255)

    print("[INFO] Loading MobileNet SSD model...")
    net = cv2.dnn.readNetFromCaffe('ssd_files/MobileNetSSD_deploy.prototxt', 'ssd_files/MobileNetSSD_deploy.caffemodel')

    if use_gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    print("[INFO] Accessing video stream...")
    vs = cv2.VideoCapture(0) if live_video else cv2.VideoCapture('test1.mp4')

    recognized_plates = []

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # ------------ 1) Object Detection -------------
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        collision_risk = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_level:
                continue

            idx = int(detections[0, 0, i, 1])
            label_name = CLASSES[idx]

            if label_name == "person" and confidence < person_confidence_threshold:
                continue  # Skip low-confidence pedestrian detections

            if label_name not in ["car", "bus", "truck", "motorbike", "person"]:
                continue  # Only process relevant classes

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            box_color = LIGHT_GREEN
            label_text = f"{label_name}: {confidence * 100:.2f}%"

            box_width = endX - startX
            distance = estimate_distance(box_width)
            if distance is not None:
                label_text += f" | Dist: {distance:.2f}m"
                if distance < COLLISION_THRESHOLD:
                    collision_risk = True
                    box_color = RED

                    if label_name in ["car", "truck", "bus"]:
                        vehicle_roi = frame[startY:endY, startX:endX].copy()
                        plate_text = detect_and_read_plate(vehicle_roi)
                        if plate_text:
                            label_text += f" | Plate: {plate_text}"
                            print(f"[INFO] Detected Plate: {plate_text}")
                            recognized_plates.append(plate_text)
                            # Send plate info to RPI
                            client_socket.sendall(f"Detected Plate: {plate_text}\n".encode())

            cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)
            label_y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label_text, (startX, label_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, box_color, 2)

        if collision_risk:
            warning_text = "COLLISION WARNING"
            font_scale = 1.5
            thickness = 3
            text_size, _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
            text_width, text_height = text_size
            x_coord = int((w - text_width) / 2)
            y_coord = 60
            cv2.putText(frame, warning_text, (x_coord, y_coord), cv2.FONT_HERSHEY_DUPLEX, font_scale, RED, thickness)
            client_socket.sendall("COLLISION WARNING\n".encode())  # Send collision warning to RPI

        # ------------ 2) Lane Detection -------------
        frame_with_lanes, avg_angle = process_lane(frame)

        if avg_angle > 5:
            curvature_text = f"Curving to the RIGHT. Angle: {avg_angle:.2f}°"
        elif avg_angle < -5:
            curvature_text = f"Curving to the LEFT. Angle: {avg_angle:.2f}°"
        else:
            curvature_text = "The road is STRAIGHT"

        print(f"[INFO] {curvature_text}")
        client_socket.sendall(f"{curvature_text}\n".encode())  # Send curvature info to RPI

        cv2.putText(frame_with_lanes, curvature_text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Object, Lane, & Plate Detection', frame_with_lanes)

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
            client_socket.sendall(f"Session Plate #{i}: {plate}\n".encode())  # Send final plate info to RPI

    client_socket.close()


if __name__ == "__main__":
    main()
