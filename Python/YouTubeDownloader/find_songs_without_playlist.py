import os

def get_all_files(directory):
    """Get all files in the given directory."""
    all_files = set()
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.add(os.path.join(root, file).replace('/', '\\'))
    return all_files

def get_files_from_playlists(directory):
    """Extract all files listed in .m3u playlists in the given directory."""
    playlist_files = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.m3u'):
                print(f"Processing playlist: {file}")
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            print(f"\t add file: {line}")
                            playlist_files.add(os.path.abspath(line).replace('/', '\\'))
    return playlist_files

def write_missing_files_to_txt(missing_files, output_file):
    """Write the list of missing files to a .txt file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in sorted(missing_files):
            f.write(file + '\n')

def main():
    song_directory = r'C:/Users/Usuario/Music/Musica JB/archivos de musica'
    playlist_directory = r'C:/Users/Usuario/Music/Musica JB/itunes_playslist_for_android/in'

    # Ensure the output file is saved in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_directory, 'missing_songs.txt')

    # Get all files and playlist files
    all_files = get_all_files(song_directory)
    playlist_files = get_files_from_playlists(playlist_directory)

    # Find missing files
    missing_files = all_files - playlist_files

    # Write missing files to a txt file
    write_missing_files_to_txt(missing_files, output_file)

    print(f"Missing files written to: {output_file}")

if __name__ == '__main__':
    main()
