import os

data_path = 'DATAPATH/5' #Folder containing data

class_to_be_changed = {
    1: 0, # old class: new class
    2: 0,
    3: 0,
    9: 0,
    10: 0,
}

if not os.path.isdir(data_path):
    raise FileNotFoundError(f"Directory not found: {data_path}")

modified_count = 0

for root, dirs, files in os.walk(data_path):
    for file in files:
        if not file.endswith(".txt") or ('README' in file):
            continue

        label_path = os.path.join(root, file)

        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        changed = False

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            class_id = int(parts[0])

            if class_id in class_to_be_changed:
                parts[0] = str(class_to_be_changed[class_id])
                changed = True

            new_lines.append(" ".join(parts) + "\n")

        # Save changes if any
        if changed:
            with open(label_path, "w") as f:
                f.writelines(new_lines)
            modified_count += 1

print(f"Modified {modified_count} label files with class changes: {class_to_be_changed}")
