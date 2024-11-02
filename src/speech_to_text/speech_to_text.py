from pydub import AudioSegment
import speech_recognition as sr
import os

def convert_to_wav(input_audio_path, output_audio_path):
    """
    Converts an audio file to .wav format using pydub.
    
    :param input_audio_path: Path to the input audio file (e.g., .mp3, .mp4)
    :param output_audio_path: Path to save the converted .wav file
    """
    audio = AudioSegment.from_file(input_audio_path)
    audio.export(output_audio_path, format="wav")
    print(f"Converted {input_audio_path} to {output_audio_path}")

def convert_speech_to_text(audio_path):
    """
    Converts speech in an audio file (in .wav format) to text using Google Web Speech API.
    
    :param audio_path: Path to the .wav audio file.
    :return: Transcribed text from the audio.
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
             text = recognizer.recognize_google(audio_data)
             print("Text:", text)
        except sr.UnknownValueError:
              print("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
              print(f"Could not request results from Google Speech Recognition service; {e}")

    return text

# Paths
input_audio_path = r'audio\audio.mp3'  # Path to the original audio file (e.g., .mp3)
output_wav_path = os.path.join('audio', 'audio.wav')  # Path where the .wav file will be saved

# Ensure the audio directory exists
if not os.path.exists('audio'):
    os.makedirs('audio')

# Step 1: Convert audio to .wav format
convert_to_wav(input_audio_path, output_wav_path)

# Step 2: Convert the .wav file to text
text = convert_speech_to_text(output_wav_path)
print("Transcribed Text:\n", text)