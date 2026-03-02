import os
import cv2
import numpy as np

# ---- Paths ----
base_dir = "FOLDER PATH"
images_dir = os.path.join(base_dir, "images")
pos_dir = os.path.join(base_dir, "positive")
neg_dir = os.path.join(base_dir, "negative")

# ---- Get base image names ----
image_files = [f for f in os.listdir(images_dir)
               if f.lower().endswith((".jpg", ".png", ".jpeg"))]

image_files.sort()
index = 0


def load_related_images(image_name):
    base = os.path.splitext(image_name)[0]

    original = cv2.imread(os.path.join(images_dir, image_name))
    if original is None:
        return None

    positives = []
    negatives = []

    for f in os.listdir(pos_dir):
        if f.startswith(base + "_pos_"):
            img = cv2.imread(os.path.join(pos_dir, f))
            if img is not None:
                positives.append(img)

    for f in os.listdir(neg_dir):
        if f.startswith(base + "_neg_"):
            img = cv2.imread(os.path.join(neg_dir, f))
            if img is not None:
                negatives.append(img)

    return original, positives, negatives


def make_grid(original, positives, negatives):
    cell_size = 256

    # resize original
    orig_resized = cv2.resize(original, (cell_size, cell_size))

    # limit number shown per row (for display)
    max_show = 4
    positives = positives[:max_show]
    negatives = negatives[:max_show]

    # pad if needed
    while len(positives) < max_show:
        positives.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

    while len(negatives) < max_show:
        negatives.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))

    positives = [cv2.resize(p, (cell_size, cell_size)) for p in positives]
    negatives = [cv2.resize(n, (cell_size, cell_size)) for n in negatives]

    row1 = np.hstack([orig_resized] + positives)
    row2 = np.hstack([np.zeros_like(orig_resized)] + negatives)

    grid = np.vstack([row1, row2])
    return grid


while True:

    image_name = image_files[index]
    data = load_related_images(image_name)

    if data is None:
        continue

    original, positives, negatives = data
    grid = make_grid(original, positives, negatives)

    cv2.imshow("Dataset Viewer (Left/Right arrows)", grid)

    key = cv2.waitKey(0)

    # Right arrow OR D
    if key == 83 or key == ord('d'):
        index = (index + 1) % len(image_files)

    # Left arrow OR A
    elif key == 81 or key == ord('a'):
        index = (index - 1) % len(image_files)

    # ESC to quit
    elif key == 27:
        break

cv2.destroyAllWindows()