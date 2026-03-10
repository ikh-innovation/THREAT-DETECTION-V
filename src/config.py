'''
PROJECT CONFIG
'''
project_config = {
    'name':'v0', # change this on new run :)
    'path':'./data',
    'runs':'./runs'
}

project_classification_config = {
    'name':'v0', # change this on new run :)
    'path':'./data',
    'runs':'./classifier_runs'
}

'''
TRAINING & INFERENCE
'''
training_config = {
    'model':'yolov8n.pt',
    'epochs': 250,
    'batch': 16,
    'size': 640
}

classification_training_config = {
    'model':'yolov8n-cls.pt',
    'epochs': 100,
    'batch': 16,
    'size': 256
}

inference_config = {
    'weights':f"./runs/{project_config['name']}/weights/best.pt",
    'confidence': 0.4,
    'classifier_confidence': 0.9,
    'test_images':'./data/test/images',
    'size':640
}

inference_classification_config = {
    'weights':f"./runs/classify/classifier_runs/{project_classification_config['name']}/weights/best.pt",
    'confidence': 0.5,
}