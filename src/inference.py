import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append('./')

from src.config import inference_config


class YoloInferencer:
    def __init__(self, path_weights: str, path_video: str, conf: float):
        self.weights = path_weights
        self.model = YOLO(self.weights)
        self.video_path = path_video
        self.conf = conf

    def run_video_inference(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video {self.video_path}")
            return

        print("Press 'q' to quit video inference.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=0.8,
                max_det=4,
                verbose=False
            )

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO Real-Time Video Inference", cv2.resize(annotated_frame, (512,512)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    path_video = "./data_ex/2.mp4"

    yolo_inferencer = YoloInferencer(
        path_weights=inference_config['weights'],
        path_video=path_video,
        conf=inference_config['confidence']
    )

    yolo_inferencer.run_video_inference()
