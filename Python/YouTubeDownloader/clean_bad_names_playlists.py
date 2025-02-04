import os
import re

# Define the directory containing the .m3u files
playlist_directory = r'C:\Users\Usuario\Music\Musica JB\itunes_playslist_for_android\in'

# Regex pattern to match files starting with two digits and a blank space
pattern = re.compile(r'^\d{2} ')

# Iterate through all .m3u files in the directory
for playlist_file in os.listdir(playlist_directory):
    if playlist_file.endswith('.m3u'):
        playlist_path = os.path.join(playlist_directory, playlist_file)

        # Temporary storage for modified playlist contents
        updated_lines = []

        print(f"Processing playlist: {playlist_file}")

        # Open and read the playlist with a tolerant encoding
        with open(playlist_path, 'r', encoding='ISO-8859-1', errors='replace') as file:
            for line in file:
                line = line.strip()
                # Skip comments or empty lines
                if not line or line.startswith('#'):
                    updated_lines.append(line)
                    continue

                # Check if the file starts with two digits and a space
                base_name = os.path.basename(line)
                dir_name = os.path.dirname(line)

                if pattern.match(base_name):
                    # Remove the two digits and space
                    new_base_name = pattern.sub('', base_name)
                    new_line = os.path.join(dir_name, new_base_name)

                    # Rename the file on disk if it exists
                    old_file_path = os.path.join(dir_name, base_name)
                    new_file_path = os.path.join(dir_name, new_base_name)

                    if os.path.exists(old_file_path):
                        os.rename(old_file_path, new_file_path)
                        print(f"Renamed: {old_file_path} -> {new_file_path}")

                    # Use the updated path in the playlist
                    updated_lines.append(new_line)
                else:
                    updated_lines.append(line)

        # Overwrite the playlist with updated paths
        with open(playlist_path, 'w', encoding='ISO-8859-1', errors='replace') as file:
            file.write('\n'.join(updated_lines) + '\n')

print("All playlists processed.")
