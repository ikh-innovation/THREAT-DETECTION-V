import os
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'DATAPATH/1' #Folder containing data

image_extensions = ["jpg", "jpeg", "png"]  # Supported image formats

if not os.path.isdir(data_path):
    raise FileNotFoundError(f"Directory not found: {data_path}")

image_count = 0
dataset_classes = {}
label_files_count = 0

for root, dirs, files in os.walk(data_path):
    for file in files:

        if file.endswith(tuple(image_extensions)):
            image_count+=1

        if not file.endswith(".txt") or ('README' in file):
            continue
        
        label_files_count += 1
        label_path = os.path.join(root, file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            label_class = line.strip().split()[0]
            dataset_classes[label_class]  = dataset_classes.get(label_class, 0) + 1

dataset_classes = dict(sorted(dataset_classes.items()))
print(f'Number of images: {image_count}')
print('Number of label files:', label_files_count)
print('Available classes:', dataset_classes)

keys = list(dataset_classes.keys())
values = list(dataset_classes.values())

plt.figure(figsize=(8, 5))
sns.barplot(x=keys, y=values, palette="viridis")

plt.title('Available classes')
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()