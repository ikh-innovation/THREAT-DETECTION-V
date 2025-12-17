from roboflow import Roboflow
rf = Roboflow(api_key="E4P1asOlS3HJ2EwCHUOy")
project = rf.workspace("magisterka-gdfg0").project("military_footage_recognition")
version = project.version(5)
dataset = version.download("yolov8")