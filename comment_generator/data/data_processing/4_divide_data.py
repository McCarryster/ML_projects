import json
import os

data_dir = '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/json_data.json'
output_dir = '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data'

with open(data_dir, 'r') as file:
    data = json.load(file)

total_elements = len(data)
training_size = 700000
evaluation_size = 50000
testing_size = total_elements - training_size - evaluation_size

training_set = data[:training_size]
evaluation_set = data[training_size:training_size + evaluation_size]
testing_set = data[training_size + evaluation_size:]

print(f"Training set length: {len(training_set)}")
print(f"Evaluation set length: {len(evaluation_set)}")
print(f"Testing set length: {len(testing_set)}")


with open(os.path.join(output_dir, 'training_set.json'), 'w') as file:
    json.dump(training_set, file, indent=4)

with open(os.path.join(output_dir, 'evaluation_set.json'), 'w') as file:
    json.dump(evaluation_set, file, indent=4)

with open(os.path.join(output_dir, 'testing_set.json'), 'w') as file:
    json.dump(testing_set, file, indent=4)