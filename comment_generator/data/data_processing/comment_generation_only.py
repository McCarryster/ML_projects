import json
import os

train_set = '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/training_set.json'
val_set = '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/evaluation_set.json'
test_set = '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/testing_set.json'


def filter_json(data_set, output_dir):
    with open(data_set, 'r') as file:
        data = json.load(file)
    
    filtered_data = [
        obj for obj in data if obj["input"]["task"] == "comment_generation"
    ]

    with open(output_dir, 'w') as json_file:
        json.dump(filtered_data, json_file, indent=4)

    print(f"Filtered {len(filtered_data)} objects with 'comment_generation' task.")
    # Filtered 366579 objects with 'comment_generation' task.
    # Filtered 30755 objects with 'comment_generation' task.
    # Filtered 29091 objects with 'comment_generation' task.


filter_json(train_set, '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/comment_only_train_set.json')
filter_json(val_set, '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/comment_only_val_set.json')
filter_json(test_set, '/media/mccarryster/stuff/very_big_work/ML_projects/comments_reply_generator/data/data_types/processed_data/comment_only_test_set.json')