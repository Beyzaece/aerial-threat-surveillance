from pathlib import Path
from PIL import Image

# VisDrone -> bizim sınıflar
# 1 pedestrian -> 0 person
# 2 people     -> 0 person
# 4 car        -> 1 car
# 6 truck      -> 2 truck

CLASS_MAP = {
    1: 0,
    2: 0,
    4: 1,
    6: 2,
}


def convert_split(images_dir, annotations_dir, labels_dir):
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    labels_dir = Path(labels_dir)

    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*.jpg"))

    for img_path in image_files:
        ann_path = annotations_dir / f"{img_path.stem}.txt"
        out_path = labels_dir / f"{img_path.stem}.txt"

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        yolo_lines = []

        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 8:
                        continue

                    x, y, w, h = map(int, parts[:4])
                    category = int(parts[5])

                    if category not in CLASS_MAP:
                        continue

                    if w <= 0 or h <= 0:
                        continue

                    class_id = CLASS_MAP[category]

                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    norm_w = w / img_w
                    norm_h = h / img_h

                    yolo_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                    )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    convert_split(
        images_dir="dataset/images/train",
        annotations_dir="train_annotations",
        labels_dir="dataset/labels/train"
    )

    convert_split(
        images_dir="dataset/images/val",
        annotations_dir="val_annotations",
        labels_dir="dataset/labels/val"
    )

    print("Dönüşüm tamamlandı.")