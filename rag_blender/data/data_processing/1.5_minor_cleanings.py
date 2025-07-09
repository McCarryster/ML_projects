import os

root_dir = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_blender/data/data_types/processed_data/cleaned_html"

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "copyright_cleaned.html":
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
