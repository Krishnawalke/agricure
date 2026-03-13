import json
import os

dataset_path = './dataset/plantvillage/color'
classes = sorted(os.listdir(dataset_path))
print(f'Found {len(classes)} classes')
print(classes)

with open('classes.json', 'w') as f:
    json.dump(classes, f)
print('classes.json saved!')