import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from spellchecker import SpellChecker
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import whisper
from keybert import KeyBERT
import os
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
# Import the necessary functions
from moviepy.config import change_settings
from moviepy.editor import TextClip, CompositeVideoClip

# Set the path for ImageMagick here
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

# Create directories if they don't exist
if not os.path.exists('text'):
    os.makedirs('text')




# Create the output directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Download the model using snapshot_download
model_path = snapshot_download(
    repo_id="sentence-transformers/distilbert-base-nli-mean-tokens",
    revision="main",
    resume_download=True
)

# Load transformer-based models
punctuation_corrector = pipeline("text2text-generation", model="facebook/bart-large")
spell_checker = SpellChecker()
kw_model = KeyBERT(model_path)

# Load the pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Define the max input length for BART (typically 1024 tokens)
MAX_INPUT_LENGTH = 1024

def correct_punctuation_bart(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=False)
    input_ids = inputs["input_ids"][0]
    
    # Split input_ids into chunks
    chunks = [input_ids[i:i + MAX_INPUT_LENGTH] for i in range(0, len(input_ids), MAX_INPUT_LENGTH)]
    corrected_chunks = []
    
    for chunk in chunks:
        chunk = torch.unsqueeze(chunk, 0)
        summary_ids = model.generate(chunk, max_length=MAX_INPUT_LENGTH, num_beams=4, early_stopping=True)
        corrected_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        corrected_chunks.append(corrected_text)
    
    return " ".join(corrected_chunks)

def correct_spelling(text):
    corrected_words = []
    for word in text.split():
        correction = spell_checker.correction(word)
        corrected_words.append(correction if correction else word)
    return " ".join(corrected_words)

def remove_repetitive_statements(text):
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = []
    previous_sentence = ""
    for sentence in sentences:
        if sentence != previous_sentence:
            cleaned_sentences.append(sentence)
        previous_sentence = sentence
    return " ".join(cleaned_sentences)

def extract_entities(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)
    entities = []
    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            entity = " ".join([leaf[0] for leaf in subtree.leaves()])
            entity_type = subtree.label()
            entities.append((entity, entity_type))
    return entities

def segment_paragraphs_keybert(text):
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    dictionary = Dictionary(tokenized_sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in tokenized_sentences]

    lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    topic_sentences = [[] for _ in range(5)]

    for i, sentence in enumerate(corpus):
        topic_distribution = lda.get_document_topics(sentence)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        topic_sentences[dominant_topic].append(sentences[i])

    paragraphs_with_topics = []
    for topic_id, sentences in enumerate(topic_sentences):
        if sentences:
            paragraph_text = " ".join(sentences)
            keywords = kw_model.extract_keywords(paragraph_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=1)
            topic_name = f"Topic {topic_id + 1}: {keywords[0][0]}" if keywords else f"Topic {topic_id + 1}: (Unnamed)"
            paragraphs_with_topics.append((topic_name, paragraph_text))

    return paragraphs_with_topics

def sync_with_timestamps(whisper_result, processed_text, paragraphs_with_topics):
    timestamp_data = whisper_result['segments']
    total_audio_duration = timestamp_data[-1]['end']
    total_text_length = sum(len(paragraph[1]) for paragraph in paragraphs_with_topics)

    synced_data = []
    for i, (topic_name, paragraph) in enumerate(paragraphs_with_topics):
        start_time = sum(len(paragraphs_with_topics[j][1]) for j in range(i)) / total_text_length * total_audio_duration
        end_time = sum(len(paragraphs_with_topics[j][1]) for j in range(i + 1)) / total_text_length * total_audio_duration

        synced_data.append({
            'topic': topic_name,
            'text': paragraph,
            'start': start_time,
            'end': end_time
        })

    return synced_data

"""def overlay_text_on_video(video_path, synced_data, output_path):
    clip = VideoFileClip(video_path)
    clips = []

    for item in synced_data:
        txt_clip = TextClip(item['text'], fontsize=24, color='white', bg_color='black', size=clip.size)
        txt_clip = txt_clip.set_pos('center').set_duration(item['end'] - item['start']).set_start(item['start'])
        clips.append(txt_clip)

    video_with_text = CompositeVideoClip([clip] + clips)
    video_with_text.write_videofile(output_path, codec='libx264', audio_codec='aac')  """


def overlay_youtube_style_timestamps(video_path, synced_data, output_path):
    """Overlays topic and timestamps on the video according to the synced timestamps."""
    try:
        # Load the video file and retrieve properties
        clip = VideoFileClip(video_path)
        width, height = clip.size
        fps = clip.fps

        clips = [clip]

        for item in synced_data:
            start_time_str = f"{int(item['start'] // 60):02}:{int(item['start'] % 60):02}"
            end_time_str = f"{int(item['end'] // 60):02}:{int(item['end'] % 60):02}"
            text = f"{item['topic']} [{start_time_str} - {end_time_str}]"

            # Adjust fontsize dynamically based on video resolution
            fontsize = max(24, int(width * 0.03))  # For example, 3% of the width

            txt_clip = (TextClip(text, fontsize=fontsize, color='white', font='Arial')
                        .set_position(('center', height - int(height * 0.1)))  # Slightly above bottom edge
                        .set_duration(item['end'] - item['start'])
                        .set_start(item['start']))

            clips.append(txt_clip)

        # Composite video and set audio to original audio
        video_with_text = CompositeVideoClip(clips)
        video_with_text.audio = clip.audio

        # Write the final video, using original fps
        video_with_text.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=fps)
        print(f"Video with timestamps created successfully: {output_path}")
    except Exception as e:
        print(f"Error overlaying text on video: {e}")






def process_text(whisper_result):
    raw_text = ' '.join([segment['text'] for segment in whisper_result['segments']])
    print("Raw Whisper Transcription:", raw_text)
    
    punctuated_text = correct_punctuation_bart(raw_text)
    print("After Punctuation Correction:", punctuated_text)
    
    corrected_text = correct_spelling(punctuated_text)
    cleaned_text = remove_repetitive_statements(corrected_text)
    print("After Spelling and Repetitive Statements Removal:", cleaned_text)
    
    entities = extract_entities(cleaned_text)
    print("Named Entities:", entities)
    
    paragraphs = segment_paragraphs_keybert(cleaned_text)
    print("Segmented Paragraphs:", paragraphs)
    
    synced_output = sync_with_timestamps(whisper_result, cleaned_text, paragraphs)
    print("Synced Data with Topics:", synced_output)

    return {
        'paragraphs': paragraphs,
        'synced_data': synced_output,
        'entities': entities
    }

# Example usage
def generate_text(video_path, output_path,audio_path):
    whisper_model = whisper.load_model("base")
    whisper_result = whisper_model.transcribe(audio_path)  # Replace with your audio file
    

    #here we have to use   output path instead of text/output.txt 
    output = process_text(whisper_result)
    with open(output_path, 'w') as file:
        file.write(f"Paragraphs:\n")
        for paragraph in output['paragraphs']:
            file.write(f"{paragraph}\n\n")

        file.write(f"Synced Timestamps with Text:\n")
        for item in output['synced_data']:
            file.write(f"Topic: {item['topic']}, Text: {item['text']}, Start: {item['start']:.2f}, End: {item['end']:.2f}\n")

        file.write(f"Named Entities:\n")
        for entity in output['entities']:
            file.write(f"Entity: {entity[0]}, Type: {entity[1]}\n")
    return output


# Overlay text on video
# Call the function
def overlay_timestamps(video_path, output_path,ouput):
    output=output

    overlay_youtube_style_timestamps(video_path, output['synced_data'],output_path)

    #here we have to place output path instaed of static one 

#overlay_text_on_video("video/video.mp4", output['synced_data'], "output/output_with_text.mp4")  # Replace with your video file

# Store output in files
"""with open('text/output.txt', 'w') as file:
    file.write(f"Paragraphs:\n")
    for paragraph in output['paragraphs']:
        file.write(f"{paragraph}\n\n")

    file.write(f"Synced Timestamps with Text:\n")
    for item in output['synced_data']:
        file.write(f"Topic: {item['topic']}, Text: {item['text']}, Start: {item['start']:.2f}, End: {item['end']:.2f}\n")

    file.write(f"Named Entities:\n")
    for entity in output['entities']:
        file.write(f"Entity: {entity[0]}, Type: {entity[1]}\n")"""

"""# Print output
print("Paragraphs:")
for paragraph in output['paragraphs']:
    print(paragraph)

print("\nSynced Timestamps with Topics:")
for item in output['synced_data']:
    print(f"Topic: {item['topic']}, Text: {item['text']}, Start: {item['start']:.2f}, End: {item['end']:.2f}")

print("\nNamed Entities:")
for entity in output['entities']:
    print(f"Entity: {entity[0]}, Type: {entity[1]}")"""
