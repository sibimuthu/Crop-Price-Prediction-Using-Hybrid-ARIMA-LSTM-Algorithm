from rembg import remove
from PIL import Image
import numpy as np
import os
import sys

def process_image(input_path, output_dir):
    print(f"Processing {input_path}...")
    try:
        # 1. Load Image
        img = Image.open(input_path).convert("RGBA")
        
        # 2. Remove Background
        print("Removing background...")
        img_no_bg = remove(img)
        
        # 3. Find Contours/Split
        # Convert to numpy array to find non-transparent pixels
        arr = np.array(img_no_bg)
        alpha = arr[:, :, 3]
        
        # Find horizontal projections (sum of alpha values along columns)
        # This helps find gaps between objects
        col_sums = np.sum(alpha, axis=0)
        
        # Identify start/end of objects based on alpha presence
        has_content = col_sums > 0
        
        # Find continuous segments of content
        segments = []
        in_segment = False
        start = 0
        for i, val in enumerate(has_content):
            if val and not in_segment:
                start = i
                in_segment = True
            elif not val and in_segment:
                segments.append((start, i))
                in_segment = False
        if in_segment:
            segments.append((start, len(has_content)))
            
        print(f"Found {len(segments)} segments.")
        
        # Map segments to produce names (assuming order: Tomato, Potato, Onion based on user prompt sequence)
        names = ["packet_tomato.png", "packet_potato.png", "packet_onion.png"]
        
        # Fallback: if segments != 3, just split into thirds
        if len(segments) != 3:
            print("Warning: Did not find exactly 3 segments. Splitting into thirds.")
            width, height = img.size
            segment_width = width // 3
            segments = [
                (0, segment_width), 
                (segment_width, segment_width*2), 
                (segment_width*2, width)
            ]

        # 4. Save Segments
        for i, (start, end) in enumerate(segments):
            if i >= len(names): break
            
            # Crop with some padding if possible, or exact
            segment_img = img_no_bg.crop((start, 0, end, img.size[1]))
            
            # Trim transparent borders from the crop
            bbox = segment_img.getbbox()
            if bbox:
                segment_img = segment_img.crop(bbox)
            
            save_path = os.path.join(output_dir, names[i])
            segment_img.save(save_path)
            print(f"Saved {names[i]} to {save_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Input file from brain folder
    brain_dir = r"c:/Users/Sibiraj/.gemini/antigravity/brain/aedfe86d-ad04-4ff4-b750-b4a720e6668f"
    input_file = os.path.join(brain_dir, "uploaded_media_1770537554413.png")
    
    # Output to static folder
    output_folder = r"c:/Users/Sibiraj/Desktop/Final year proj/static"
    
    if os.path.exists(input_file):
        process_image(input_file, output_folder)
    else:
        print(f"Input file not found: {input_file}")
