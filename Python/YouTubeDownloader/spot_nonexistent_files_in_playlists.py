import os
import unicodedata
import chardet

# Define the directory containing the .m3u files
playlist_directory = r'C:\Users\Usuario\Music\Musica JB\itunes_playslist_for_android\in'

# Dictionary to keep track of missing files per playlist
missing_files = {}

# Get the directory where the script is saved
script_directory = os.path.dirname(os.path.abspath(__file__))

# Output file to save results
output_file = os.path.join(script_directory, 'missing_files_report.txt')

# Normalize paths for consistent comparison
def normalize_path(path):
    return unicodedata.normalize('NFC', path)

# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

# Iterate through all .m3u files in the directory
for playlist_file in os.listdir(playlist_directory):
    if playlist_file.endswith('.m3u'):
        playlist_path = os.path.join(playlist_directory, playlist_file)

        # Detect the encoding of the file
        file_encoding = detect_encoding(playlist_path)

        # Initialize a list to track missing files for this playlist
        missing_files[playlist_file] = []

        # Open and read the playlist with the detected encoding
        with open(playlist_path, 'r', encoding=file_encoding, errors='replace') as file:
            for line in file:
                line = line.strip()
                # Skip comments or empty lines
                if not line or line.startswith('#'):
                    continue

                # Normalize the path for comparison
                normalized_line = normalize_path(line)

                # Check if the file exists
                if not os.path.exists(normalized_line):
                    missing_files[playlist_file].append(normalized_line)

# Write the results to the output file
with open(output_file, 'w', encoding='utf-8') as out_file:
    for playlist, missing in missing_files.items():
        playlist_full_path = os.path.join(playlist_directory, playlist)
        out_file.write(f"\nPlaylist: {playlist}\n")
        out_file.write(f"Full Path: {playlist_full_path}\n")
        if missing:
            out_file.write("Missing files:\n")
            for file in missing:
                # Normalize the file path for consistent representation
                normalized_missing_file = normalize_path(file)
                out_file.write(f"  {normalized_missing_file}\n")
        else:
            out_file.write("All files exist.\n")

print(f"Missing files report saved to: {output_file}")
