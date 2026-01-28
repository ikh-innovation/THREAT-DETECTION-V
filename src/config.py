'''
PROJECT CONFIG
'''
project_config = {
    'name':'v0', # change this on new run :)
    'path':'./data',
    'runs':'./runs'
}
'''
TRAINING & INFERENCE
'''
training_config = {
    'model':'yolov8n.pt',
    'epochs': 100,
    'batch': 8,
    'size': 640
}

inference_config = {
    'weights':f"./runs/{project_config['name']}/weights/best.pt",
    'confidence': 0.6,
    'test_images':'./data/test/images',
    'size':640
}