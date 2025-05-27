import json
import os
from collections import Counter
import matplotlib.pyplot as plt


data_dir = '/home/mccarryster/very_big_work_ubuntu/ML_projects/comment_generator/data/data_types/processed_data/comment_only_test_set.json'
output_dir = '/home/mccarryster/very_big_work_ubuntu/ML_projects/comment_generator/data/data_types/processed_data/filtered'
with open(data_dir, 'r') as file:
    data = json.load(file)

# Step 1: Get the length of each output string
output_lengths = [len(item['output']) for item in data]

# Step 2: Count the occurrences of each length
length_distribution = Counter(output_lengths)

# Step 3: Filter data points where output length is less than 50
filtered_data = [item for item in data if len(item['output']) >= 50]

# Step 4: Filter data points where the length occurrence is less than 50
filtered_data = [item for item in filtered_data if length_distribution[len(item['output'])] >= 50]

# Save the filtered data back to the file (optional)
with open(os.path.join(output_dir, 'comment_only_test_set.json'), 'w') as file:
    json.dump(filtered_data, file, indent=4)

print(f"Original data size: {len(data)}")
print(f"Filtered data size: {len(filtered_data)}")