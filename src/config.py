'''
PROJECT CONFIG
'''
project_config = {
    'name':'runlolol',
    'path':'./data',
    'runs':'./runs'
}
'''
TRAINING & INFERENCE
'''
training_config = {
    'model':'yolov8s.pt',
    'epochs': 150,
    'batch': 8,
    'size': 2
}

inference_config = {
    'weights':'./runs/threat_detection/weights/best.pt',
    'confidence': 0.5,
    'test_images':'./data/test/images',
    'size':2
}