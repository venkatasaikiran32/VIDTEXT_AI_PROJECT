from moviepy.editor import VideoFileClip
import os

AUDIO = 'audio'

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    # Ensure the audio is saved with a valid extension, e.g., .mp3 or .wav
    audio.write_audiofile(audio_path)

# Create the audio directory if it doesn't exist
if not os.path.exists(AUDIO):
    os.makedirs(AUDIO)

# Sample checking: Extract audio and save as 'audio.mp3'
#extract_audio(r'uploads\ntr.mp4', os.path.join(AUDIO, 'audio.mp3'))
