import json
import os

script_dir = os.path.dirname(__file__)  # This gets the directory where see_data.py is located
json_file_path = os.path.join(script_dir, '../data_types/processed_data/ready_data.json')  # Construct the full path

with open(json_file_path, 'r') as file:
    data = json.load(file)

for data_point in data:
    print(data_point)
    break