import os
import sys
from ultralytics import YOLO

sys.path.append('./')

from src.config import project_config, training_config

class YoloTrainer:

    def __init__(self, path_dataset:str, path_yaml:str):
        '''
        Yolo trainer initialization
        
        :param path_dataset: dataset path in yolov8 format
        :param path_yaml: yaml file of yolov8 dataset
        :param model_name: name of yolv8 model
        '''
        self.model_name = training_config['model']
        self.model = YOLO(self.model_name)

        self.dataset =  path_dataset
        self.yaml = os.path.join(self.dataset, path_yaml)
        pass

    def train(self):

        self.model.train(
        data=self.yaml,
        epochs=training_config['epochs'],
        batch=training_config['batch'],
        imgsz=training_config['size'],
        project= project_config['runs'],
        name=project_config['name'],
        exist_ok=True,
        device=0
        )

if __name__ == "__main__":

    path_dataset = './military_dataset'
    path_yaml = 'data.yaml'

    
    yolo_trainer = YoloTrainer(path_dataset= path_dataset,
                               path_yaml= path_yaml)
    yolo_trainer.train()

