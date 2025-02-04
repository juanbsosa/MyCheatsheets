import os
import sys
import glob
import time

def create_recent_songs_playlist(song_directory, playlist_directory):
    """
    Creates an .m3u playlist of the 100 most recently modified audio files.

    Args:
        song_directory (str): The directory containing the audio files.
        playlist_directory (str): The directory where the .m3u playlist will be saved.

    Returns:
        None
    """

    # Define the audio file extensions to look for
    audio_extensions = ('*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg', '*.m4a', '*.wma')

    # Get all audio files in the song_directory
    audio_files = []
    for extension in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(song_directory, '**', extension), recursive=True))

    if not audio_files:
        print("No audio files found in the specified directory.")
        return

    # Get the 100 most recently modified audio files
    audio_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    recent_files = audio_files[:100]

    # Ask the user for the playlist name
    playlist_name = input("Enter the name for the new playlist (without extension): ").strip()
    if not playlist_name:
        print("Playlist name cannot be empty.")
        return

    # Create the full path for the .m3u file
    playlist_path = os.path.join(playlist_directory, playlist_name + '.m3u')

    # Write the absolute paths to the .m3u file
    with open(playlist_path, 'w', encoding='utf-8') as playlist_file:
        for file_path in recent_files:
            playlist_file.write(file_path + '\n')

    print(f"Playlist '{playlist_name}.m3u' has been created with {len(recent_files)} songs.")

if __name__ == '__main__':
    song_directory = input("Enter the path to the song directory: ").strip()
    playlist_directory = input("Enter the path to the playlist directory: ").strip()

    if not os.path.isdir(song_directory):
        print("The song directory does not exist.")
        sys.exit(1)
    if not os.path.isdir(playlist_directory):
        print("The playlist directory does not exist.")
        sys.exit(1)

    create_recent_songs_playlist(song_directory, playlist_directory)