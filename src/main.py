import cv2
import argparse
from pathlib import Path
from ultralytics import RTDETR

roi_points = []
drawing = False
temp_point = None


def draw_roi(event, x, y, flags, param):
    global roi_points, drawing, temp_point

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True
        temp_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing = False
        temp_point = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    return parser.parse_args()


def get_risk_level(class_name):
    class_name = class_name.lower()

    if class_name in ["drone", "helicopter"]:
        return "HIGH", (0, 0, 255)
    elif class_name == "bird":
        return "LOW", (0, 255, 0)
    elif class_name in ["car", "truck", "person"]:
        return "MEDIUM", (0, 255, 255)
    else:
        return "UNKNOWN", (255, 255, 255)


def class_conf_threshold(class_name):
    class_name = class_name.lower()

    thresholds = {
        "person": 0.40,
        "car": 0.40,
        "truck": 0.45,
        "bird": 0.55,
        "drone": 0.65,
        "helicopter": 0.60,  
    }
    return thresholds.get(class_name, 0.50)


def is_inside_roi(cx, cy, roi):
    (x1, y1), (x2, y2) = roi

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    return left <= cx <= right and top <= cy <= bottom


def draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, corner_len=18):
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)

    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)

    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)

    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_label_box(img, text_lines, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    pad = 8
    line_h = 22

    max_width = 0
    for line in text_lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)

    box_w = max_width + pad * 2
    box_h = line_h * len(text_lines) + pad

    x = max(5, x)
    if x + box_w > img.shape[1] - 5:
        x = img.shape[1] - box_w - 5

    y = max(y, box_h + 5)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - box_h), (x + box_w, y), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.rectangle(img, (x, y - box_h), (x + box_w, y), color, 1)

    for i, line in enumerate(text_lines):
        ty = y - box_h + pad + 16 + i * line_h
        cv2.putText(img, line, (x + pad, ty), font, font_scale, color, thickness)


def draw_roi_overlay(img, roi):
    (x1, y1), (x2, y2) = roi

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

    draw_corner_box(img, left, top, right, bottom, (255, 0, 0), thickness=2, corner_len=22)

    draw_label_box(
        img,
        ["RESTRICTED ZONE", "SECURITY PERIMETER"],
        left,
        max(top - 8, 60),
        (255, 0, 0)
    )


def draw_header(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 48), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.putText(
        img,
        "AI CRITICAL AREA SURVEILLANCE SYSTEM",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "RT-DETR ENGINE",
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1
    )


def draw_footer_info(img, image_mode=False):
    text = "Mouse: Draw ROI   |   C: Clear ROI   |   Q: Quit"
    if image_mode:
        text = "Mouse: Draw ROI   |   C: Clear ROI   |   S: Save   |   Q: Quit"

    cv2.putText(
        img,
        text,
        (10, img.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def load_image_safely(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None

    # PNG alpha kanalı varsa 4 -> 3 kanal
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # grayscale ise 3 kanala çevir
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def process_frame(frame, model):
    global roi_points, drawing, temp_point

    clean_frame = frame.copy()
    results = model(clean_frame, verbose=False)
    r = results[0]

    display_frame = frame.copy()
    alert_text = None
    highest_alert_priority = -1

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, box)
            class_name = names[cls[i]]
            
            min_conf = class_conf_threshold(class_name)

            if conf[i] < min_conf:
                continue

            risk, color = get_risk_level(class_name)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            draw_corner_box(display_frame, x1, y1, x2, y2, color, thickness=2, corner_len=16)

            info_lines = [
                f"{class_name.upper()} | {conf[i]:.2f}",
                f"THREAT: {risk}"
            ]
            draw_label_box(display_frame, info_lines, x1, max(y1 - 10, 60), color)

            if len(roi_points) == 2 and is_inside_roi(cx, cy, roi_points):
                draw_corner_box(display_frame, x1, y1, x2, y2, (0, 0, 255), thickness=3, corner_len=22)

                current_alert = None
                current_priority = -1

                if class_name.lower() in ["drone", "helicopter"]:
                    current_alert = "CRITICAL THREAT IN RESTRICTED ZONE"
                    current_priority = 3
                elif class_name.lower() in ["car", "truck", "person"]:
                    current_alert = "INTRUSION ALERT"
                    current_priority = 2
                elif class_name.lower() == "bird":
                    current_alert = "LOW RISK OBJECT IN RESTRICTED ZONE"
                    current_priority = 1

                if current_priority > highest_alert_priority:
                    highest_alert_priority = current_priority
                    alert_text = current_alert

    if len(roi_points) == 2:
        draw_roi_overlay(display_frame, roi_points)

    elif drawing and len(roi_points) == 1 and temp_point is not None:
        x1, y1 = roi_points[0]
        x2, y2 = temp_point
        draw_corner_box(
            display_frame,
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
            (255, 0, 0),
            thickness=2,
            corner_len=20
        )

    draw_header(display_frame)

    if alert_text:
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 58), (720, 105), (0, 0, 120), -1)
        cv2.addWeighted(overlay, 0.55, display_frame, 0.45, 0, display_frame)

        cv2.putText(
            display_frame,
            alert_text,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return display_frame


def main():
    global roi_points, drawing, temp_point

    args = parse_args()
    model = RTDETR("models/best.pt")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source_path = Path(args.source)

    cv2.namedWindow("Threat Surveillance")
    cv2.setMouseCallback("Threat Surveillance", draw_roi)

    # FOTOĞRAF MODU
    if source_path.suffix.lower() in image_extensions:
        frame = load_image_safely(args.source)

        if frame is None:
            print("Görsel açılamadı")
            return

        while True:
            display_frame = process_frame(frame, model)
            draw_footer_info(display_frame, image_mode=True)

            cv2.imshow("Threat Surveillance", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                roi_points = []
                drawing = False
                temp_point = None
            elif key == ord("s"):
                cv2.imwrite("output_detection.png", display_frame)
                print("Kaydedildi: output_detection.png")

    
    else:
        if args.source.isdigit():
            cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(args.source)

        if not cap.isOpened():
            print("Video açılamadı")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame alınamadı")
                break

            display_frame = process_frame(frame, model)
            draw_footer_info(display_frame, image_mode=False)

            cv2.imshow("Threat Surveillance", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                roi_points = []
                drawing = False
                temp_point = None

        cap.release()

    cv2.destroyAllWindows()
    print("Sistem kapatıldı")


if __name__ == "__main__":
    main()