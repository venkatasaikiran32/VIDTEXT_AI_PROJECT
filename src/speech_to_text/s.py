import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe("audio/audio.mp3")

# Create the 'text' folder if it doesn't exist
if not os.path.exists('text'):
    os.makedirs('text')

# Extract the transcribed text from the result and write it to a file
with open('text/raw_text.txt', 'w') as file:
    file.write(result['text'])  # Write only the transcribed text



#whisper is an open ai library that can translate the speech to text
# even it can capitalize and add punctuations
# as it is trained with noise data also , so it can translate even some what noise is present
# as it rturns the result in dict
#to get transcribed result we use result['text']

#features
#it has some what does punctuations, and some what contextual based auto correction
#furthur if we need post preprocessing we have to do it, using other librarries