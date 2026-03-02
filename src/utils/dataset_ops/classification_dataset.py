import os
import cv2
import random

data_path = 'FOLDER PATH'
image_extensions = {".jpg", ".jpeg", ".png"}

if not os.path.isdir(data_path):
    raise FileNotFoundError(f"Directory not found: {data_path}")


def overlap(box1, box2, iou_threshold=0.1):
    x_left = max(box1["x_min"], box2["x_min"])
    y_top = max(box1["y_min"], box2["y_min"])
    x_right = min(box1["x_max"], box2["x_max"])
    y_bottom = min(box1["y_max"], box2["y_max"])

    if x_right <= x_left or y_bottom <= y_top:
        return False

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area1 = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
    area2 = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])

    union = area1 + area2 - intersection

    if union <= 0:
        return False

    iou = intersection / union
    
    return iou > iou_threshold


for root, dirs, files in os.walk(data_path):
    for file in files:
        if not file.endswith(".txt") or ('README' in file):
            continue

        label_path = os.path.join(root, file)

        image_name = os.path.splitext(file)[0]
        image_path = None

        for ext in image_extensions:
            candidate = os.path.join(root, image_name + ext).replace('labels', 'images')
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print("No image found for:", file)
            continue

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Failed to load image. Unsupported format or corrupted file.")

        img_height, img_width = img.shape[:2]

        labels_dimensions = []
        counter = 1

        with open(label_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                parts = [float(x) for x in line.strip().split()]
                class_id = int(parts[0])

                if class_id == 0:

                    x_c_pixel = parts[1] * img_width
                    y_c_pixel = parts[2] * img_height
                    w_pixel   = parts[3] * img_width
                    h_pixel   = parts[4] * img_height

                    x_min = x_c_pixel - w_pixel / 2
                    y_min = y_c_pixel - h_pixel / 2
                    x_max = x_c_pixel + w_pixel / 2
                    y_max = y_c_pixel + h_pixel / 2

                    pad_w = 0.2 * w_pixel
                    pad_h = 0.2 * h_pixel

                    x_min -= pad_w
                    y_min -= pad_h
                    x_max += pad_w
                    y_max += pad_h

                    x_min = max(0, int(x_min))
                    y_min = max(0, int(y_min))
                    x_max = min(img_width, int(x_max))
                    y_max = min(img_height, int(y_max))

                    labels_dimensions.append({
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    })

                    min_crop = 5
                    if (x_max - x_min) < min_crop:
                        x_max = min(img_width, x_min + min_crop)
                    if (y_max - y_min) < min_crop:
                        y_max = min(img_height, y_min + min_crop)

                    crop = img[y_min:y_max, x_min:x_max]
                    crop = cv2.resize(crop, (256, 256))
                    parent_dir = os.path.dirname(root)
                    output_dir = os.path.join(parent_dir, 'positive')
                    os.makedirs(output_dir, exist_ok=True)

                    output_path = os.path.join(
                        output_dir,
                        f"{image_name}_pos_{counter}.jpg"
                    )
                    cv2.imwrite(output_path, crop)

                    counter += 1

        num_random_crops = len(labels_dimensions)

        for i in range(1, num_random_crops + 1):
            attempts = 0

            while attempts < 100:
                if img_width <= 200 or img_height <= 200:
                    # small image: enforce minimum crop size
                    min_size = max(10, min(img_width, img_height) // 4)
                    crop_w = random.randint(min_size, max(min_size, img_width // 4))
                    crop_h = random.randint(min_size, max(min_size, img_height // 4))
                else:
                    # normal image: keep previous logic
                    crop_w = random.randint(50, img_width // 2)
                    crop_h = random.randint(50, img_height // 2)

                x_min = random.randint(0, img_width - crop_w)
                y_min = random.randint(0, img_height - crop_h)
                x_max = x_min + crop_w
                y_max = y_min + crop_h

                random_box = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                }

                overlap_found = False
                for label_box in labels_dimensions:
                    if overlap(random_box, label_box):
                        overlap_found = True
                        break

                if not overlap_found:
                    crop = img[y_min:y_max, x_min:x_max]
                    crop = cv2.resize(crop, (256, 256))
                    output_dir = os.path.join(parent_dir, 'negative')
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(
                        output_dir,
                        f"{image_name}_neg_{i}.jpg"
                    )
                    cv2.imwrite(output_path, crop)
                    break

                attempts += 1

            if attempts == 50:
                print('Attemp:', i ,'Image negative failed:',image_name)