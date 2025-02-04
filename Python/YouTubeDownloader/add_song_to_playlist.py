import os

def add_missing_songs_to_playlists(missing_songs_file, playlist_directory):
    """
    Go through the missing songs one by one, ask the user to add them to a playlist,
    and add the song to the selected playlists.

    Args:
        missing_songs_file (str): Path to the file containing missing songs.
        playlist_directory (str): Path to the directory containing playlists.

    Returns:
        None
    """

    def list_playlists(directory):
        """List all playlists (.m3u files) in the given directory."""
        playlists = [file for file in os.listdir(directory) if file.endswith('.m3u')]
        return playlists

    def add_to_playlist(playlist_path, song):
        """Append a song to the specified playlist."""
        with open(playlist_path, 'a', encoding='utf-8') as f:
            f.write(song + '\n')
        print(f"Added {song} to playlist {os.path.basename(playlist_path)}.")

    # Read missing songs from the file
    with open(missing_songs_file, 'r', encoding='utf-8') as f:
        missing_songs = [line.strip() for line in f if line.strip()]

    for song in missing_songs:
        print(f"Processing: {song}")
        playlists = list_playlists(playlist_directory)

        if not playlists:
            print("No playlists found in the directory.")
            return

        print("Available playlists:")
        for i, playlist in enumerate(playlists, start=1):
            print(f"{i}. {playlist}")

        while True:
            choice = input("Enter playlist numbers (comma-separated), 'new' to create a new playlist, or press enter to skip this song: ").lower()
            if choice == '':
                print(f"Skipped adding {song} to any playlist.")
                break
            elif choice == 'new':
                new_playlist_name = input("Enter the name for the new playlist: ").strip()
                if new_playlist_name:
                    new_playlist_path = os.path.join(playlist_directory, new_playlist_name + '.m3u')
                    open(new_playlist_path, 'w').close()
                    print(f"Created new playlist: {new_playlist_name}.m3u")
                    add_to_playlist(new_playlist_path, song)
                else:
                    print("Invalid playlist name. Try again.")
            else:
                try:
                    selected_indices = [int(num.strip()) for num in choice.split(',')]
                    for index in selected_indices:
                        if 1 <= index <= len(playlists):
                            playlist_path = os.path.join(playlist_directory, playlists[index - 1])
                            add_to_playlist(playlist_path, song)
                        else:
                            print(f"Invalid playlist number: {index}")
                    break
                except ValueError:
                    print("Invalid input. Please enter numbers, 'new', or 'skip'.")

if __name__ == '__main__':
    missing_songs_file = r"C:\\Users\\Usuario\\Documents\\Notas-de-Computacion\\Python\\YouTubeDownloader\\missing_songs.txt"
    playlist_directory = r"C:\\Users\\Usuario\\Music\\Musica JB\\itunes_playslist_for_android\\in"
    add_missing_songs_to_playlists(missing_songs_file, playlist_directory)
