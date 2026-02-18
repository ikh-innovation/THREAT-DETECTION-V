import os
import time
import random
import cv2
import torch
from ultralytics import YOLO

class YoloONNXBenchmark:
    def __init__(self, onnx_path, images_path):
        self.onnx_path = onnx_path
        self.images_path = images_path

        print(f"Loading ONNX model: {onnx_path}")
        self.model = YOLO(onnx_path)

        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print("Running ONNX GPU inference 🚀")

        self.class_thresholds = {
            0: 0.30, 
            1: 0.50  
        }

    def warmup(self, img, runs=5):
        """Warm up GPU for accurate benchmarking"""
        print("Warming up ONNX model...")
        for _ in range(runs):
            _ = self.model.predict(
                source=img,
                imgsz=640,
                device=self.device,
                verbose=False
            )
        torch.cuda.synchronize()
        print("Warmup complete.\n")

    def filter_boxes_by_class(self, results):
        """Apply class-specific confidence thresholds correctly"""
        boxes = results[0].boxes
        scores = boxes.conf
        classes = boxes.cls

        keep_mask = []
        for i, cls_id in enumerate(classes):
            conf_threshold = self.class_thresholds.get(int(cls_id), 0.25)  
            keep_mask.append(scores[i] >= conf_threshold)

        keep_mask = torch.tensor(keep_mask, dtype=torch.bool)

        filtered_boxes = boxes[keep_mask]

        results[0].boxes = filtered_boxes  
        return results


    def benchmark(self):
        """Run inference on all images and measure FPS"""
        image_files = [
            os.path.join(self.images_path, f)
            for f in os.listdir(self.images_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        if not image_files:
            print("No images found in folder.")
            return

        random.shuffle(image_files)

        img = cv2.imread(image_files[0])
        self.warmup(img)

        total_time = 0
        count = 0

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read: {img_path}")
                continue

            torch.cuda.synchronize()
            start = time.perf_counter()

            results = self.model.predict(
                source=img,
                imgsz=640,
                device=self.device,
                verbose=False
            )

            results = self.filter_boxes_by_class(results)

            torch.cuda.synchronize()
            end = time.perf_counter()

            inference_time = end - start
            total_time += inference_time
            count += 1

            fps = 1.0 / inference_time
            print(f"[{count}] {os.path.basename(img_path)} "
                  f"| {inference_time*1000:.2f} ms "
                  f"| {fps:.2f} FPS "
                  f"| Boxes kept: {len(results[0].boxes)}")

            annotated = results[0].plot()
            cv2.imshow("YOLO ONNX Inference", cv2.resize(annotated, (512,512)))
            cv2.waitKey(0)

        if count > 0:
            avg_time = total_time / count
            print("\n========== FINAL RESULTS (ONNX GPU) ==========")
            print(f"Images processed: {count}")
            print(f"Average time per image: {avg_time*1000:.2f} ms")
            print(f"Average FPS: {1.0/avg_time:.2f}")
            print("==============================================")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    images_path = "./data/valid/images"  
    onnx_model_path = "runs/v0/weights/best.onnx"

    benchmark = YoloONNXBenchmark(
        onnx_path=onnx_model_path,
        images_path=images_path
    )
    benchmark.benchmark()
