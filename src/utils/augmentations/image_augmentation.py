import sys
import os
import cv2
import numpy as np
sys.path.append('./')

class ImageAugmentation:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor
        
    def adjust_brightness(self, image):
        """Increase brightness by multiplying pixel values."""

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * self.brightness_factor
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def rotate_90(self, image):
        """Rotate image exactly 90 degrees clockwise."""
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def apply_augmentation(self, img_path, mode):
        """
        Load an image from a path and apply augmentation.
        
        Args:
            img_path (str): Path to the image file.
            mode (str): "rotate90", "brightness", or "default".
        
        Returns:
            numpy.ndarray: Augmented image (BGR format).
        """
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if mode == "rotate90":
            return self.rotate_90(img)
        elif mode == "brightness":
            return self.adjust_brightness(img)


# Example usage
if __name__ == "__main__":
    augmentor = ImageAugmentation(brightness_factor=1.5)
    img_path = "./datasets/train\images\-_-_-_-_-_-_-_-_-_-_-_mp4-3_jpg.rf.496565309287c296fede91cf98d85e1f.jpg"

    rotated_img = augmentor.apply_augmentation(img_path, mode="rotate90")
    bright_img = augmentor.apply_augmentation(img_path, mode="brightness")

    cv2.imshow("Original", cv2.imread(img_path))
    cv2.imshow("Rotated 90°", rotated_img)
    cv2.imshow("Brightness", bright_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

