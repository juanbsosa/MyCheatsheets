import subprocess
import sys
import pkg_resources
from pkg_resources import parse_version



def install(package, upgrade=False, from_github=False):
    if from_github:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/pytube/pytube"])
    elif upgrade:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_and_update_pytube():
    try:
        import pytube
        installed_version = pytube.__version__

        # Get the latest version from PyPI
        import requests
        response = requests.get('https://pypi.org/pypi/pytube/json', timeout=10)
        latest_version = response.json()['info']['version']

        if parse_version(installed_version) < parse_version(latest_version):
            print(f"Updating pytube from version {installed_version} to {latest_version}...")
            install('pytube', upgrade=True)
        else:
            print(f"pytube is up to date (version {installed_version}).")
    except ImportError:
        print("pytube is not installed. Installing from GitHub...")
        install('pytube', from_github=True)
    except Exception as e:
        print(f"An error occurred while checking pytube version: {e}")
        print("Attempting to install pytube from GitHub...")
        install('pytube', from_github=True)


required_packages = ['ffmpeg', 'yt-dlp', 'mutagen', 'requests']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package '{package}' is not installed. Installing...")
        install(package)

# After ensuring all packages are installed, check and update pytube
check_and_update_pytube()

from pytube import YouTube
import os
import sys
# from mutagen.easyid3 import EasyID3
# from mutagen.mp3 import MP3
from mutagen.mp4 import MP4

# def download_audio(url, dest_dir, playlist_dir):
    # """
    # Download the audio from a YouTube video and save it as an MP3 file.

    # Args:
    #     url (str): The URL of the YouTube video to download.
    #     dest_dir (str): The destination directory to save the MP3 file.
    #                     Defaults to the current directory.

    # Returns:
    #     str: The full path to the downloaded MP3 file.

    # """

    # while True:
    #     # Create a YouTube object
    #     try:
    #         # yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    #         yt = YouTube(url)
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         break

    #     # Extract audio from video
    #     audio = yt.streams.filter(only_audio=True).first()

    #     # Download the audio file to dest_dir
    #     out_file = audio.download(output_path=dest_dir)

    #     # Split path into base and extension, to convert to .mp3
    #     base, ext = os.path.splitext(out_file)

    #     # Option to define a new file name
    #     new_title = input("Enter new file name in the format Artist - Song Name: ")
    #     if new_title:
    #         base = os.path.join(os.path.dirname(base), new_title)

    #     # # Change to .mp3
    #     # new_file = base + '.mp3'

    #     # Change to .mp4
    #     new_file = base + '.mp4'

    #     # Rename downloaded file
    #     os.rename(out_file, new_file)

    #     # Set Title and Artist (if allowed)
    #     # !!! not working properly
    #     # try:
    #     Artist, Title  = new_title.split(" - ")
    #     # mpfile = MP3(new_file, ID3=EasyID3)
    #     mpfile = MP4(new_file)
    #     # mpfile['title'] = Title
    #     # mpfile['artist'] = Artist
    #     mpfile['\xa9nam'] = Title
    #     mpfile['\xa9ART'] = Artist
    #     mpfile.save()
    #     # except:
    #     #     pass

    #     # Report success
    #     print(f"{yt.title} has been successfully downloaded to {new_file}")
    #     print("\n")

    #     # Ask whether to add the audio file to a playlist
    #     answer = input("Do you want to add the audio file to a playlist? (y/n): ")
    #     if answer.lower() == "y":
    #         # if playlist_dir is None:
    #         #     playlist_dir = input("Please provide the directory where the playlists are stored: ")
    #         search_playlists(playlist_dir, new_file)

    #     # Prompt to download another video
    #     download_another = input("Do you want to download audio from another video? (y/n): ")
    #     if download_another.lower() == 'y':
    #         url = input("Enter video URL: ")
    #     else:
    #         break

def download_audio(url, dest_dir, playlist_dir):
    """
    Download the audio from a YouTube video and save it as an MP3 file.
    """
    
    while True:
        from yt_dlp import YoutubeDL

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(dest_dir, '%(title)s.%(ext)s'),
            'ffmpeg_location': r'C:\ffmpeg\bin',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            }

        with YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(url, download=True)
                title = info_dict.get('title', None)
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        # Get the downloaded file path
        filename = ydl.prepare_filename(info_dict)
        base, ext = os.path.splitext(filename)
        mp3_file = base + '.mp3'

        # Rename and tag the file
        new_title = input("Enter new file name in the format Artist - Song Name: ")
        if new_title:
            new_file = os.path.join(os.path.dirname(mp3_file), new_title + '.mp3')
            os.rename(mp3_file, new_file)
        else:
            new_file = mp3_file

        # Set Title and Artist using mutagen
        try:
            Artist, Title = new_title.split(" - ")
            from mutagen.mp3 import MP3
            from mutagen.id3 import ID3, TIT2, TPE1

            audio = MP3(new_file, ID3=ID3)
            audio.tags.add(TIT2(encoding=3, text=Title))
            audio.tags.add(TPE1(encoding=3, text=Artist))
            audio.save()
        except Exception as e:
            print(f"Failed to set metadata: {e}")

        print(f"{title} has been successfully downloaded to {new_file}\n")

        # # Ask whether to add the audio file to a playlist
        # answer = input("Do you want to add the audio file to a playlist? (y/n): ")
        # if answer.lower() == "y":
        #     search_playlists(playlist_dir, new_file)

        # Ask whether to add the audio file to a playlist
        while True:
            answer = input("Do you want to add the audio file to a playlist? (yes/no/new): ").lower()
            if answer in ["yes", "y"]:
                search_playlists(playlist_dir, new_file)
                break
            elif answer in ["no", "n"]:
                print("Audio file not added to any playlist.")
                break
            elif answer == "new":
                create_new_playlist(playlist_dir)
                # After creating a new playlist, the loop continues
                # User will be asked again if they want to add the file to a playlist
            else:
                print("Invalid input. Please enter 'yes', 'no', or 'new'.")


        # Prompt to download another video
        download_another = input("Do you want to download audio from another video? (y/n): ")
        if download_another.lower() == 'y':
            url = input("Enter video URL: ")
        else:
            break


def search_playlists(playlist_directory, audio_file_directory):
    """
    Search for playlists in the provided directory and prompt the user to select playlists
    to which the audio file should be moved.

    Args:
        playlist_directory (str): The directory where the playlists are stored.
        audio_file_directory (str): The directory of the downloaded audio file.

    Returns:
        None
    """

    playlists = []
    for file in os.listdir(playlist_directory):
        if file.endswith(".m3u"):
            playlists.append(file)

    print("Playlists in the provided directory:")
    for i, playlist in enumerate(playlists, start=1):
            print(f"{i}. {playlist}")
        
    playlist_numbers = input("Enter the playlist number (or numbers seprated by a comma) where you want to move the audio file, or enter 'no' to not move it to any playlist: ")
    
    if playlist_numbers == 'no':
        print("Audio file not moved to any playlist.")
    else:
        try:
            selected_playlists = [int(num.strip()) for num in str(playlist_numbers).split(",")]
            for playlist_number in selected_playlists:
                if 1 <= playlist_number <= len(playlists):
                    selected_playlist = playlists[playlist_number - 1]
                    add_to_playlist(playlist_directory, selected_playlist, audio_file_directory)
                else:
                    print("Invalid playlist number!")
        except ValueError:
            print("Invalid playlist number!")


def add_to_playlist(playlist_directory, playlist, audio_file_directory):
    """
    Appends the path of the downloaded audio file to the selected playlist.

    Args:
        playlist_directory (str): The directory where the playlists are stored.
        playlist (str): The name of the selected playlist.
        audio_file_directory (str): The directory of the downloaded audio file.

    Returns:
        None
    """

    playlist_path = os.path.join(playlist_directory, playlist)

    # Append the path of the downloaded audio file to the selected playlist
    audio_path = os.path.join(audio_file_directory)
    with open(playlist_path, "a") as file:
        file.write(audio_path + "\n")

    print("Audio file added to the playlist:", playlist)


def create_new_playlist(playlist_directory):
    """
    Prompts the user to create a new playlist and creates an empty .m3u file.

    Args:
        playlist_directory (str): The directory where the playlists are stored.

    Returns:
        None
    """
    while True:
        playlist_name = input("Enter the name of the new playlist (without extension): ").strip()
        if not playlist_name:
            print("Playlist name cannot be empty. Please try again.")
            continue
        playlist_path = os.path.join(playlist_directory, playlist_name + '.m3u')
        if os.path.exists(playlist_path):
            print("A playlist with that name already exists. Please choose a different name.")
            continue
        else:
            # Create an empty .m3u file
            open(playlist_path, 'w').close()
            print(f"New playlist '{playlist_name}.m3u' created successfully.")
            break

    # Ask if the user wants to create another new playlist
    while True:
        create_another = input("Do you want to create another new playlist? (yes/no): ").lower()
        if create_another in ["yes", "y"]:
            create_new_playlist(playlist_directory)
            break  # Exit the current loop after recursive call
        elif create_another in ["no", "n"]:
            print("Returning to playlist selection...")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


if __name__ == '__main__':
    url = input("Enter video URL: ")
    dest_dir = input("Enter destination directory or leave blank for default directory: ") or r"C:\Users\Usuario\Music\Musica JB\archivos de musica"
    playlist_dir = input("Enter playlist directory or leave blank for default directory: ") or r"C:\Users\Usuario\Music\MusicBee\Playlists"

    file_path = download_audio(url, dest_dir, playlist_dir)

    if not file_path:
        sys.exit(1)
