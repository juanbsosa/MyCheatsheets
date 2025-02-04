import os

# Define the directory to search
base_directory = r'C:\Users\Usuario\Music\Musica JB\archivos de musica'

# Initialize a counter for deleted files
deleted_files_count = 0

# Iterate through all files in the directory
for file_name in os.listdir(base_directory):
    # Check if the file name starts with two numbers followed by a space
    if len(file_name) > 3 and file_name[:2].isdigit() and file_name[2] == ' ':
        # Create the corresponding file name without the two numbers and space
        trimmed_file_name = file_name[3:]

        # Check if the corresponding file exists in the directory
        trimmed_file_path = os.path.join(base_directory, trimmed_file_name)
        original_file_path = os.path.join(base_directory, file_name)

        if os.path.exists(trimmed_file_path):
            try:
                # Delete the file that starts with two numbers and a space
                os.remove(original_file_path)
                deleted_files_count += 1
                print(f"Deleted: {original_file_path}")
            except Exception as e:
                print(f"Error deleting file {original_file_path}: {e}")
        else:
            print(f"No duplicate found for: {original_file_path}")

print(f"Task completed. Total files deleted: {deleted_files_count}")