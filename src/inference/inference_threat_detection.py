import sys
import cv2
from ultralytics import YOLO

sys.path.append('./')
from src.config import inference_config, inference_classification_config


class ThreatDetection:
    def __init__(self, obj_detection_path_weights: str, classifier_path_weights: str, path_image: str, confidence: float, classifier_confidence: float, use_classifier: bool, visualize: bool):
        self.obj_detection_model = YOLO(obj_detection_path_weights)
        self.classifier_model = YOLO(classifier_path_weights)
        self.image_path = path_image
        self.confidence = confidence
        self.classifier_confidence = classifier_confidence
        self.use_classifier = use_classifier
        self.visualize = visualize

    def vehicle_classifier(self, img):
        if img is None:
            raise RuntimeError("Failed to load image. Unsupported format or corrupted file.")

        results = self.classifier_model.predict(
            source=img,
            imgsz=256,
            verbose=False
        )
        r = results[0]

        probs = r.probs
        label = self.classifier_model.names[int(probs.top1)]

        return label == 'positive'
    
    def crop_img(self, img, box_coordinates):
        if img is None:
            raise RuntimeError("Failed to load image. Unsupported format or corrupted file.")

        x_min, y_min, x_max, y_max = box_coordinates
        img_height, img_width = img.shape[:2]

        w_pixel   = x_max-x_min
        h_pixel   = y_max-y_min

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

        min_crop = 5
        if (x_max - x_min) < min_crop:
            x_max = min(img_width, x_min + min_crop)
        if (y_max - y_min) < min_crop:
            y_max = min(img_height, y_min + min_crop)

        crop = img[y_min:y_max, x_min:x_max]
        cropped_img = cv2.resize(crop, (256, 256))

        return cropped_img
    
    def display_detection(self, img, box):

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_name = self.obj_detection_model.names[int(box.cls[0])]  

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Threat detect Inference", cv2.resize(img, (512, 512)))

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        

    def run_detect_threat(self):

        img = cv2.imread(self.image_path)
        if img is None:
            print("Could not read:", self.image_path)
            return

        results = self.obj_detection_model.predict(
            source=img,
            conf=self.confidence,
            iou=0.1,
            imgsz = 640,
            verbose=False
        )

        for result in results:
            boxes = result.boxes  # Bounding boxes
            if len(boxes)==0: print('No object detection')
            
            for box in boxes:
                detected = False
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Pixel coordinates
                conf = float(box.conf[0])              # Confidence score
                cls_id = int(box.cls[0])               # Class ID
                label = self.obj_detection_model.names[cls_id]            # Class name
   
                if conf <= self.classifier_confidence and self.use_classifier:
                    cropped_img = self.crop_img(img, box.xyxy[0].tolist())
                    detected = self.vehicle_classifier(cropped_img)
                else:
                    detected = True
                if detected:
                    print(f"{label}: Confidence {conf:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    print('No object detection')

                if self.visualize:
                    frame = img.copy()
                    self.display_detection(frame, box)

if __name__ == "__main__":

    path_image = "INPUT PATH"

    yolo_inferencer = ThreatDetection(
        obj_detection_path_weights=inference_config['weights'],
        classifier_path_weights=inference_classification_config['weights'],
        path_image=path_image,
        confidence=inference_config['confidence'],
        classifier_confidence=inference_config['classifier_confidence'],
        use_classifier=True,
        visualize=True
    )

    yolo_inferencer.run_detect_threat()
