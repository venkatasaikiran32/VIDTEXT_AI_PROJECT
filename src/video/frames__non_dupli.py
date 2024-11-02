import cv2  # OpenCV library for image processing
import os   # To work with file paths
from skimage.metrics import structural_similarity as ssim  # To compare images

# Load all extracted frames
def load_frames(folder):
    # This function loads each frame from a folder
    frames = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))  # Load image
        if img is not None:  # Make sure image is loaded
            frames.append(img)  # Add image to list of frames
    return frames

# Compare two frames to see if they are duplicates
def is_duplicate(frame1, frame2, threshold=0.95):
    # Convert images to grayscale (simplifies comparison)
    grayA = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compare the two frames using SSIM (a similarity measure)
    score, _ = ssim(grayA, grayB, full=True)
    return score > threshold  # If similarity is higher than threshold, it's a duplicate

# Extract unique frames by removing duplicates
def extract_unique_frames(frames, threshold=0.95):
    unique_frames = [frames[0]]  # Keep the first frame
    for i in range(1, len(frames)):
        if not is_duplicate(frames[i-1], frames[i], threshold):
            unique_frames.append(frames[i])  # Save unique frame
    return unique_frames

# Save the unique frames to a folder
def save_frames(frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the folder if it doesn't exist
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_folder, f"unique_frame_{i}.png"), frame)  # Save each frame

# Main code to run everything
frames_folder = 'frames'  # Path where your frames are saved
output_folder = 'non_dupli_frames'  # Path to save the unique frames

# Load all frames from the folder
frames = load_frames(frames_folder)

# Get the unique frames by comparing them
unique_frames = extract_unique_frames(frames)

# Save the unique frames to a folder
save_frames(unique_frames, output_folder)

print(f"Saved {len(unique_frames)} unique frames.")
