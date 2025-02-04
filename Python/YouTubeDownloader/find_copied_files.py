import os
import re

def find_duplicate_files(directory):
    # Dictionary to store filenames and their occurrences
    file_groups = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Use a regular expression to normalize filenames by removing the (1), (2), etc.
        match = re.match(r"(.*?)(\(\d+\))?\.(.+)", filename)
        if match:
            base_name = match.group(1).strip()
            extension = match.group(3).strip()
            normalized_name = f"{base_name}.{extension}"

            # Group filenames by their normalized name
            if normalized_name not in file_groups:
                file_groups[normalized_name] = []
            file_groups[normalized_name].append(filename)

    # Find and print duplicate groups
    duplicates = {k: v for k, v in file_groups.items() if len(v) > 1}
    if duplicates:
        print("Duplicate files found:")
        for normalized_name, files in duplicates.items():
            print(f"Group for '{normalized_name}':")
            for file in files:
                print(f"  - {file}")
    else:
        print("No duplicate files found.")

# Replace with your directory path
directory_path = r'C:\Users\Usuario\Music\Musica JB\archivos de musica'
find_duplicate_files(directory_path)
