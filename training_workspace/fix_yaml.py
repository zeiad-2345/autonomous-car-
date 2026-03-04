import yaml
import os

yaml_path = "Bosch Traffic Signs YOLOv8/data.yaml"
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

abs_path = os.path.abspath("Bosch Traffic Signs YOLOv8")
data['train'] = os.path.join(abs_path, "train/images")
data['val'] = os.path.join(abs_path, "valid/images")
data['test'] = os.path.join(abs_path, "test/images")

with open(yaml_path, 'w') as file:
    yaml.dump(data, file)

print("Updated yaml:")
print(yaml.dump(data))
