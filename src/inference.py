import os
import sys
import random
import cv2
from ultralytics import YOLO

sys.path.append('./')
from src.config import inference_config


class YoloInferencer:
    def __init__(self, path_weights: str, path_images: str, conf: float):
        self.weights = path_weights
        self.model = YOLO(self.weights)
        self.images_path = path_images
        self.conf = conf

    def run_video_inference(self, video_path: str):
        """Run YOLO inference on a video file."""

        if not os.path.isfile(video_path):
            print(f"Error: Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video:", video_path)
            return

        print("Running video inference... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            results = self.model.predict(
                source=cv2.resize(frame, (640,640)),
                conf=self.conf,
                iou=0.8,
                max_det=10,
                verbose=False
            )

            annotated = results[0].plot()

            cv2.imshow("YOLOv8 Video Inference", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_image_folder_inference(self):

        if not os.path.isdir(self.images_path):
            print(f"Error: Folder not found: {self.images_path}")
            return

        # Collect images
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

            results = self.model.predict(
                source=img,
                conf=self.conf,
                iou=0.8,
                max_det=4,
                verbose=False
            )

            annotated = results[0].plot()

            cv2.imshow("YOLO Image Inference", cv2.resize(annotated, (512, 512)))

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_path = "./data_ext/sample_videos/2.mp4"

    yolo_inferencer = YoloInferencer(
        path_weights=inference_config['weights'],
        path_images="./data_ext/sample_dataset/test/images",
        conf=inference_config['confidence']
    )

    yolo_inferencer.run_video_inference(video_path)
    yolo_inferencer.run_image_folder_inference()
