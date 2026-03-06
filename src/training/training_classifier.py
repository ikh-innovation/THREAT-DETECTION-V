import os
import sys
from ultralytics import YOLO

sys.path.append("./")

from src.utils.training.logging_methods import *
from src.config import project_classification_config, classification_training_config


class BinaryClassifierTrainer:
    def __init__(self, path_dataset: str):
        self.last_ckpt = os.path.join(
            project_classification_config["runs"],
            project_classification_config["name"],
            "weights",
            "last.pt",
        )

        if os.path.exists(self.last_ckpt):
            print(f"🔁 Resuming training from {self.last_ckpt}")
            self.model = YOLO(self.last_ckpt)
            self.resume = True
        else:
            print("🆕 Starting new training run")
            self.model = YOLO(classification_training_config["model"])
            self.resume = False

        self.dataset_dir  = path_dataset

        self.run_dir = os.path.join(
            project_classification_config["runs"],
            project_classification_config["name"],
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
            data=self.dataset_dir,
            task='classify',
            epochs=classification_training_config["epochs"],
            batch=classification_training_config["batch"],
            imgsz=classification_training_config["size"],
            project=project_classification_config["runs"],
            name=project_classification_config["name"],
            exist_ok=True,
            device=0,
            resume=self.resume,
        )


if __name__ == "__main__":
    trainer = BinaryClassifierTrainer(
        path_dataset="./data/classification_data",
    )

    trainer.train()
