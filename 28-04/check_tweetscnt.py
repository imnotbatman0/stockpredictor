import os

def count_files_in_subfolders(root_folder):
    for subdir in os.listdir(root_folder):
        full_path = os.path.join(root_folder, subdir)
        if os.path.isdir(full_path):
            file_count = sum(
                1 for item in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, item))
            )
            print(f"{subdir}: {file_count} file(s)")

# Replace with your folder path
folder_path = "./preprocessed_data/tweets/"
count_files_in_subfolders(folder_path)
