from pytube import YouTube
import os
import sys
from tqdm import tqdm

def download_audio(url, dest_dir='.'):
    """
    Download the audio from a YouTube video and save it as an MP3 file.

    Args:
        url (str): The URL of the YouTube video to download.
        dest_dir (str): The destination directory to save the MP3 file.
                        Defaults to the current directory.

    Returns:
        str: The full path to the downloaded MP3 file.

    """

    while True:
        # Create a YouTube object
        try:
            yt = YouTube(url)
        except:
            print("Invalid URL or network error.")
            continue

        # Extract audio from video
        audio = yt.streams.filter(only_audio=True).first()

        # Download the audio file to dest_dir
        out_file = audio.download(output_path=dest_dir)

        # Split path into base and extension, to convert to .mp3
        base, ext = os.path.splitext(out_file)

        # Option to define a new file name
        new_title = input("Enter new file name in the format Artist - Song Name, or leave blank and use video's title: ")
        if new_title:
            base = os.path.join(os.path.dirname(base), new_title)

        # Change to .mp3
        new_file = base + '.mp3'

        # Rename downloaded file
        os.rename(out_file, new_file)

        # Report success
        print(f"{yt.title} has been successfully downloaded to {new_file}")
        print("\n")

        # Prompt to download another video
        download_another = input("Do you want to download audio from another video? (y/n): ")
        if download_another.lower() == 'y':
            url = input("Enter video URL: ")
        else:
            break

    return new_file

if __name__ == '__main__':
    url = input("Enter video URL: ")
    dest_dir = input("Enter destination directory or leave blank for current directory: ") or "."

    file_path = download_audio(url, dest_dir)

    if not file_path:
        sys.exit(1)
