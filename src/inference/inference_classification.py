import os
import sys
import random
import cv2
import time
import torch
from ultralytics import YOLO

sys.path.append('./')
from src.config import inference_classification_config


class YoloClassifierInferencer:
    def __init__(self, path_weights: str, path_images: str, conf: float):
        self.weights = path_weights
        self.model = YOLO(self.weights)
        self.images_path = path_images
        self.conf = conf

    def run_image_folder_inference(self):

        if not os.path.isdir(self.images_path):
            print(f"Error: Folder not found: {self.images_path}")
            return

        exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [
            os.path.join(self.images_path, f)
            for f in os.listdir(self.images_path)
            if f.lower().endswith(exts)
        ]

        if not image_files:
            print("No images found in folder:", self.images_path)
            return

        random.shuffle(image_files)
        print("Press 'q' to quit.")

        for img_path in image_files:

            img = cv2.imread(img_path)
            if img is None:
                print("Could not read:", img_path)
                continue

            start_time = time.perf_counter()

            results = self.model.predict(
                source=img,
                imgsz=256,
                verbose=False
            )

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000

            r = results[0]

            probs = r.probs
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            label = self.model.names[class_id]

            annotated = img.copy()

            label_text = f"{label}: {confidence:.2f}"
            time_text = f"Inference: {inference_time_ms:.2f} ms"

            cv2.putText(
                annotated,
                label_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                time_text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("YOLO Classification",cv2.resize(annotated, (512, 512)))

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    path_images = "./data/classification_data/test/negative"

    yolo_inferencer = YoloClassifierInferencer(
        path_weights=inference_classification_config['weights'],
        path_images=path_images,
        conf=inference_classification_config['confidence']
    )

    yolo_inferencer.run_image_folder_inference()