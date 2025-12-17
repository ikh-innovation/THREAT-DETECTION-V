'''
PROJECT CONFIG
'''
project_config = {
    'name':'threat_detection',
    'path':'./data',
    'runs':'./runs'
}
'''
TRAINING & INFERENCE
'''
training_config = {
    'model':'yolov8.pt',
    'epochs': 150,
    'batch': 16,
    'size': 640
}

inference_config = {
    'weights':'./runs/threat_detection/weights/best.pt',
    'confidence': 0.5,
    'test_images':'./data/test/images',
    'size':640
}