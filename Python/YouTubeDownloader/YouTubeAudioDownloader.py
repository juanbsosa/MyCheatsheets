# Source: https://twitter.com/clcoding/status/1561889282188386306?s=20&t=5IZq07detCvnqD_q7lwl9w

# Import necessary packages
from pytube import YouTube
import os

# Get URL imput from user
yt = YouTube(str(input("Enter video URL: \n>>")))

# Extract audio from video
audio = yt.streams.filter(only_audio=True).first()

# Enter directory
print("Enter directory or leave blank for current directory")
destination = str(input(">> ")) or "."

# Download the audio file
out_file = audio.download(output_path=destination)

# Save the file as .mp3
base, ext = os.path.splitext(out_file)
    # Option to define a file name different from the video's title
new_title = str(input("Write file name in the format Artist - Song Name, or leave blank and use video's title:")) or ""
if new_title!="":
    base = "\\".join(base.split("\\")[0:-1]) + "\\" + new_title
new_file = base + '.mp3'
os.rename(out_file, new_file)

# Report success
print(yt.title + "has been successfully downloaded.")