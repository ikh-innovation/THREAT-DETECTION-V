import os
import sys
from ultralytics import YOLO

sys.path.append("./")

from src.utils.training.logging_methods import *
from src.config import project_config, training_config


class YoloTrainer:
    def __init__(self, path_dataset: str, path_yaml: str):
        self.last_ckpt = os.path.join(
            project_config["runs"],
            project_config["name"],
            "weights",
            "last.pt",
        )

        if os.path.exists(self.last_ckpt):
            print(f"🔁 Resuming training from {self.last_ckpt}")
            self.model = YOLO(self.last_ckpt)
            self.resume = True
        else:
            print("🆕 Starting new training run")
            self.model = YOLO(training_config["model"])
            self.resume = False

        self.yaml = os.path.join(path_dataset, path_yaml)

        self.run_dir = os.path.join(
            project_config["runs"],
            project_config["name"],
        )

        self.status_dir = os.path.join(self.run_dir, "status")
        os.makedirs(self.status_dir, exist_ok=True)

        save_run_config(self.status_dir)

        self.model.add_callback(
            "on_fit_epoch_end",
            self.on_epoch_end
        )

    def on_epoch_end(self, _):
        results_csv = os.path.join(self.run_dir, "results.csv")
        if os.path.exists(results_csv):
            plot_results(results_csv, self.status_dir)

    def train(self):
        self.model.train(
            data=self.yaml,
            epochs=training_config["epochs"],
            batch=training_config["batch"],
            imgsz=training_config["size"],
            project=project_config["runs"],
            name=project_config["name"],
            exist_ok=True,
            device=0,
            resume=self.resume,
        )


if __name__ == "__main__":
    trainer = YoloTrainer(
        path_dataset="./data",
        path_yaml="data.yaml",
    )

    trainer.train()
