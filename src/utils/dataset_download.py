from roboflow import Roboflow
rf = Roboflow(api_key="E4P1asOlS3HJ2EwCHUOy")
project = rf.workspace("robert-paulson-fncbw").project("military-vehicles-detection-qwfnc")
version = project.version(2)
dataset = version.download("yolov8")