from flask import Flask, request, redirect, url_for, render_template, send_file
import os
from werkzeug.utils import secure_filename
from src.text.text_processing_actual import *
#from src.text.text_processing_actual import generate_text, overlay_timestamps  # Adjust the import according to your file structure
from src.audio.audio_extraction import *

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'video')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')

ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page for video upload
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Handle video upload
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('select_option', filename=filename))
    else:
        return "Invalid file type. Only .mp4 files are allowed."

# Page to select output options
@app.route('/options/<filename>')
def select_option(filename):
    return render_template('options.html', filename=filename)

# Generate selected output
@app.route('/generate_output', methods=['POST'])
def generate_output():
    filename = request.form['filename']
    option = request.form['option']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    output_path = ''
    
    # Execute corresponding processing based on user choice
    if option == 'audio':
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_audio.mp3')
        extract_audio(video_path, output_path)  # Call your audio extraction function
    elif option == 'text':
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_text.txt')
        output=generate_text(video_path, output_path,os.path.join(app.config['OUTPUT_FOLDER'], 'output_audio.mp3'))  # Call your text generation function
    elif option == 'video':
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_text.txt')
        output=generate_text(video_path, output_path)  # Call your text generation function
        
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_with_timestamps.mp4')
        overlay_timestamps(video_path,output, output_path)  # Call your overlay timestamps function
    
    # Check if output file was created successfully
    if os.path.exists(output_path):
        return render_template('output.html', output_file=output_path, option=option)
    else:
        return "An error occurred while processing the video."

# Download or view output
@app.route('/download/<option>/<filename>')
def download_file(option, filename):
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(output_file, as_attachment=(option != 'view'))



if __name__ == '__main__':
    app.run(debug=True)
