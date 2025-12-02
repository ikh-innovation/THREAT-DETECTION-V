import os
import yaml

class YoloDatasetInspector:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.data_yaml_path = os.path.join(dataset_dir, "data.yaml")
        self.dataset_info = {}
        self._load_data_yaml()
        self._gather_dataset_info()

    def _load_data_yaml(self):
        """Load data.yaml file to get class names and number of classes."""
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"data.yaml not found in {self.dataset_dir}")
        with open(self.data_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.dataset_info['nc'] = data.get('nc', None)
        self.dataset_info['names'] = data.get('names', [])
        self.dataset_info['paths'] = {
            'train': os.path.join(self.dataset_dir, data.get('train', 'train/images')),
            'val': os.path.join(self.dataset_dir, data.get('val', 'valid/images')),
            'test': os.path.join(self.dataset_dir, data.get('test', 'test/images')) if 'test' in data else None
        }

    def _count_images(self, path):
        """Count number of images in a folder."""
        if not path or not os.path.exists(path):
            return 0
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        return len([f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts])

    def _gather_dataset_info(self):
        """Gather info about dataset: number of images, classes, label correspondence."""
        self.dataset_info['num_images'] = {
            'train': self._count_images(self.dataset_info['paths']['train']),
            'val': self._count_images(self.dataset_info['paths']['val']),
            'test': self._count_images(self.dataset_info['paths']['test']) if self.dataset_info['paths']['test'] else 0
        }

        self.dataset_info['class_to_label'] = {name: idx for idx, name in enumerate(self.dataset_info['names'])}
        self.dataset_info['label_to_class'] = {idx: name for idx, name in enumerate(self.dataset_info['names'])}

    def print_summary(self):
        print("=== YOLO Dataset Summary ===")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Number of classes: {self.dataset_info['nc']}")
        print(f"Class names: {self.dataset_info['names']}")
        print(f"Images per split: {self.dataset_info['num_images']}")
        print("Class ↔ Label mapping:")
        for name, idx in self.dataset_info['class_to_label'].items():
            print(f"  {name} = {idx}")

    def get_info(self):
        """Return dataset info as dictionary."""
        return self.dataset_info

# Example usage:
if __name__ == "__main__":
    dataset_dir = "./data"
    inspector = YoloDatasetInspector(dataset_dir)
    inspector.print_summary()
    info = inspector.get_info()
