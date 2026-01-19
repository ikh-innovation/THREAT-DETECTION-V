import os
import cv2
import sys
import glob
import random
import numpy as np

sys.path.append('./')

class DatasetQA:
    
    def __init__(self, datasets_path: str, set_name: str, max_images: int = 100, max_cols: int = 3):
        """
        Initialize the DatasetQA operator.

        Args:
            datasets_path (str): Parent folder containing multiple datasets.
            set_name (str): Dataset subset to visualize ('train', 'val', 'test').
            max_images (int, optional): Maximum number of images to display. Defaults to 100.
            max_cols (int, optional): Maximum number of datasets per row in the visualization. Defaults to 3.
        """
        self.datasets_path = datasets_path
        self.set_name = set_name
        self.max_images = max_images
        self.max_cols = max_cols
    
    def _draw_bboxes(self, img, label_file):
        """
        Draw bounding boxes on an image based on YOLOv8-format label file.

        Args:
            img (np.ndarray): Image on which to draw bounding boxes.
            label_file (str): Path to the YOLOv8 label file.

        Returns:
            np.ndarray: Image with bounding boxes drawn.
        """
        h, w = img.shape[:2]
        if not os.path.exists(label_file):
            return img
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            
            if int(cls) == 0:
                color = (255,0,0)
            elif int(cls) == 1:
                color = (0,0,255)
            else:
                continue
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, str(int(cls)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

    def _load_images_labels(self, dataset_path):
        """
        Load image and label file pairs for a given dataset and set.

        Args:
            dataset_path (str): Path to the dataset folder.

        Returns:
            list of tuple: List of (image_path, label_path) pairs.
        """
        images_dir = os.path.join(dataset_path, self.set_name, 'images')
        labels_dir = os.path.join(dataset_path, self.set_name, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"No images found in {images_dir}")
            return []
        
        image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        label_files = [os.path.join(labels_dir, os.path.splitext(os.path.basename(f))[0]+'.txt') 
                       for f in image_files]
        return list(zip(image_files, label_files))

    def run(self):
        """
        Run the QA visualization loop.

        Iterates over the datasets, randomly selecting images, drawing bounding boxes, 
        and displaying them in a grid layout. Supports multiple rows and ESC key to exit.
        """
        datasets = [os.path.join(self.datasets_path, d) for d in os.listdir(self.datasets_path)
                    if os.path.isdir(os.path.join(self.datasets_path, d))]
        if not datasets:
            print("No datasets found in:", self.datasets_path)
            return
        
        print(f"Found datasets: {[os.path.basename(d) for d in datasets]}")
        dataset_images = [self._load_images_labels(d) for d in datasets]
        if all(len(imgs)==0 for imgs in dataset_images):
            print("No images found in any dataset for set", self.set_name)
            return
        
        num_images = min(self.max_images, max(len(imgs) for imgs in dataset_images))
        
        for _ in range(num_images):
            imgs_to_show = []
            for imgs in dataset_images:
                if len(imgs) == 0:
                    continue
                img_path, label_path = random.choice(imgs)
                img = cv2.imread(img_path)         
                img = self._draw_bboxes(img, label_path)
                imgs_to_show.append(img)
            
            if not imgs_to_show:
                continue
            
            rows = []
            for i in range(0, len(imgs_to_show), self.max_cols):
                row_imgs = imgs_to_show[i:i+self.max_cols]
                if len(row_imgs) < self.max_cols:
                    h = row_imgs[0].shape[0]
                    w = row_imgs[0].shape[1]
                    for _ in range(self.max_cols - len(row_imgs)):
                        row_imgs.append(np.zeros((h,w,3), dtype=np.uint8))
                min_height = min(img.shape[0] for img in row_imgs)
                row_imgs_resized = [cv2.resize(img, (int(img.shape[1]*min_height/img.shape[0]), min_height))
                                    for img in row_imgs]
                row_concat = cv2.hconcat(row_imgs_resized)
                rows.append(row_concat)
            
            concat_img = cv2.vconcat(rows)
            cv2.imshow('Datasets QA', concat_img)
            
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                print("ESC pressed, exiting...")
                break
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parent_path = './datasets'
    set_name = 'train'  
    qa_operator = DatasetQA(datasets_path=parent_path, set_name=set_name, max_images=100, max_cols=3)
    qa_operator.run()
