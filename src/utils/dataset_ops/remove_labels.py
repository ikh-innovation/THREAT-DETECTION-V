import os

data_path = 'DATAPATH/5' #Folder containing data

classes_to_be_removed = {4,5,6,7,8}   # Set of class IDs to remove
image_extensions = {".jpg", ".jpeg", ".png"}  # Supported image formats

if not os.path.isdir(data_path):
    raise FileNotFoundError(f"Directory not found: {data_path}")

modified_count = 0
removed_count = 0

for root, dirs, files in os.walk(data_path):
    for file in files:
        if not file.endswith(".txt") or ('README' in file):
            continue

        label_path = os.path.join(root, file)

        # Read label file
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Filter out unwanted classes
        filtered_lines = [
            line for line in lines
            if line.strip() and int(line.strip().split()[0]) not in classes_to_be_removed
        ]

        # Overwrite label file with remaining annotations
        with open(label_path, "w") as f:
            f.writelines(filtered_lines)

        # If file changed, count it
        if len(filtered_lines) != len(lines):
            modified_count += 1

        # If label file is now empty, delete it and its image
        if not filtered_lines:
            image_name = os.path.splitext(file)[0]
            image_path = None
            for ext in image_extensions:
                candidate = os.path.join(root, image_name + ext).replace('labels','images')

                if os.path.exists(candidate):
                    image_path = candidate
                    break

            os.remove(label_path)
            if image_path:
                os.remove(image_path)
            removed_count += 1

print(f"Modified {modified_count} label files (removed unwanted classes: {classes_to_be_removed}).")
print(f"Removed {removed_count} empty label files and their images.")
