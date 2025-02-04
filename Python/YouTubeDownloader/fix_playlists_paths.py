import os
import difflib
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3

# Run the script to spot nonexistent files in playlists
nonexistent_files_script = r"C:\Users\Usuario\Documents\Notas-de-Computacion\Python\YouTubeDownloader\spot_nonexistent_files_in_playlists.py"
os.system(f'python "{nonexistent_files_script}"')

# Define the path to the report file containing missing audio files
report_file_path = r"C:\Users\Usuario\Documents\Notas-de-Computacion\Python\YouTubeDownloader\missing_files_report.txt"

# List all files in the search directory
print("Listing all files in the search directory...")
all_files = [f for f in os.listdir(r'C:\Users\Usuario\Music\Musica JB\archivos de musica') if os.path.isfile(os.path.join(r'C:\Users\Usuario\Music\Musica JB\archivos de musica', f))]


def find_similar_files(search_string, search_directory, max_results=8):
    if not os.path.exists(search_directory):
        return []

    # # List all files in the search directory
    # all_files = [f for f in os.listdir(search_directory) if os.path.isfile(os.path.join(search_directory, f))]

    # Use difflib to find the closest matches
    similar_files = difflib.get_close_matches(search_string, all_files, n=max_results)

    # Add files that end with the missing file name
    similar_files.extend(
        [os.path.join(search_directory, f) for f in all_files if f.endswith(search_string)]
    )
    
    # Remove duplicates while preserving order
    seen = set()
    similar_files = [f for f in similar_files if not (f in seen or seen.add(f))]

    return [os.path.join(search_directory, f) for f in similar_files]

def extract_metadata(file_path):
    try:
        audio = MP3(file_path, ID3=EasyID3)
        title = audio.get("title", [None])[0]
        contributing_artists = audio.get("artist", [None])[0]
        return title, contributing_artists
    except Exception:
        return None, None

missing_files_by_playlist = {}
current_playlist = None

with open(report_file_path, 'r', encoding='utf-8') as report_file:
    for line in report_file:
        line = line.strip()
        if line.startswith("Playlist:"):
            current_playlist = line.replace("Playlist:", "").strip()
            missing_files_by_playlist[current_playlist] = []
        elif line.startswith("Full Path:"):
            current_playlist_path = line.replace("Full Path:", "").strip()
            missing_files_by_playlist[current_playlist] = {"path": current_playlist_path, "missing": []}
        elif line.startswith("All files exist."):
            continue
        elif line and line != "Missing files:" and current_playlist:
            missing_files_by_playlist[current_playlist]["missing"].append(line)

user_decisions = {}

for playlist, data in missing_files_by_playlist.items():
    playlist_path = data["path"]
    missing_files = data["missing"]

    if not missing_files:
        print(f"Playlist '{playlist}' has no missing files.")
        continue

    print(f"Processing playlist: {playlist}")

    for missing_file_path in missing_files:
        missing_dir = os.path.dirname(missing_file_path)
        missing_file_name = os.path.basename(missing_file_path)

        # Check if there's a saved decision for this missing file
        if missing_file_path in user_decisions:
            selected_file = user_decisions[missing_file_path]
            print(f"Automatically replacing '{missing_file_path}' with '{selected_file}' based on previous decision.")

            # Update the playlist file
            with open(playlist_path, 'r', encoding='utf-8', errors='replace') as playlist_file:
                playlist_lines = playlist_file.readlines()

            with open(playlist_path, 'w', encoding='utf-8', errors='replace') as playlist_file:
                for playlist_line in playlist_lines:
                    if missing_file_path in playlist_line:
                        updated_line = playlist_line.replace(missing_file_path, selected_file)
                        playlist_file.write(updated_line)
                        print(f"Updated playlist: {playlist}")
                    else:
                        playlist_file.write(playlist_line)
            continue

        # Find similar files in the directory
        while True:
            similar_files = find_similar_files(missing_file_name, missing_dir)
            metadata_matches = []

            for file in similar_files:
                title, _ = extract_metadata(file)
                if title and title.lower() == os.path.splitext(missing_file_name)[0].lower():
                    metadata_matches.append(file)

            if len(metadata_matches) == 1:
                selected_file = metadata_matches[0]
                user_decisions[missing_file_path] = selected_file
                print(f"Automatically selected: {selected_file} for {missing_file_path}")

                # Update the playlist file
                with open(playlist_path, 'r', encoding='utf-8', errors='replace') as playlist_file:
                    playlist_lines = playlist_file.readlines()

                with open(playlist_path, 'w', encoding='utf-8', errors='replace') as playlist_file:
                    for playlist_line in playlist_lines:
                        if missing_file_path in playlist_line:
                            updated_line = playlist_line.replace(missing_file_path, selected_file)
                            playlist_file.write(updated_line)
                            print(f"Updated playlist: {playlist}")
                        else:
                            playlist_file.write(playlist_line)
                break

            elif len(metadata_matches) > 1:
                print(f"Multiple matches found for '{missing_file_path}' (playlist: {playlist}):")
                for i, file in enumerate(metadata_matches, 1):
                    print(f"  {i}. {file}")

                user_input = input(f"Enter a number between 1 and {len(metadata_matches)} to select, enter a new search string, or press Enter to skip: ").strip()

                if user_input.isdigit() and 1 <= int(user_input) <= len(metadata_matches):
                    selected_index = int(user_input) - 1
                    selected_file = metadata_matches[selected_index]
                    user_decisions[missing_file_path] = selected_file
                    print(f"Selected: {selected_file}")

                    # Update the playlist file
                    with open(playlist_path, 'r', encoding='utf-8', errors='replace') as playlist_file:
                        playlist_lines = playlist_file.readlines()

                    with open(playlist_path, 'w', encoding='utf-8', errors='replace') as playlist_file:
                        for playlist_line in playlist_lines:
                            if missing_file_path in playlist_line:
                                updated_line = playlist_line.replace(missing_file_path, selected_file)
                                playlist_file.write(updated_line)
                                print(f"Updated playlist: {playlist}")
                            else:
                                playlist_file.write(playlist_line)
                    break

                elif user_input:
                    print(f"Searching for new matches using: '{user_input}' (playlist: {playlist})")
                    missing_file_name = user_input
                else:
                    print("Skipping file...")
                    break

            elif len(similar_files) > 0:
                print(f"No metadata matches found for '{missing_file_path}' (playlist: {playlist}), but similar files exist:")
                for i, file in enumerate(similar_files, 1):
                    print(f"  {i}. {file}")

                user_input = input(f"Enter a number between 1 and {len(similar_files)} to select, enter a new search string, or press Enter to skip: ").strip()

                if user_input.isdigit() and 1 <= int(user_input) <= len(similar_files):
                    selected_index = int(user_input) - 1
                    selected_file = similar_files[selected_index]
                    user_decisions[missing_file_path] = selected_file
                    print(f"Selected: {selected_file}")

                    # Update the playlist file
                    with open(playlist_path, 'r', encoding='utf-8', errors='replace') as playlist_file:
                        playlist_lines = playlist_file.readlines()

                    with open(playlist_path, 'w', encoding='utf-8', errors='replace') as playlist_file:
                        for playlist_line in playlist_lines:
                            if missing_file_path in playlist_line:
                                updated_line = playlist_line.replace(missing_file_path, selected_file)
                                playlist_file.write(updated_line)
                                print(f"Updated playlist: {playlist}")
                            else:
                                playlist_file.write(playlist_line)
                    break

                elif user_input:
                    print(f"Searching for new matches using: '{user_input}'")
                    missing_file_name = user_input
                else:
                    print("Skipping file...")
                    break

            else:
                print(f"No matches found for '{missing_file_path}' (playlist: {playlist}).")
                user_input = input("Enter a new search string or press Enter to skip: ").strip()

                if user_input:
                    print(f"Searching for new matches using: '{user_input}'")
                    missing_file_name = user_input
                else:
                    print("Skipping file...")
                    break

print("All playlists processed.")
