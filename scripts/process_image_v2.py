from PIL import Image
import os
import sys

def remove_bg_color(img, tolerance=30):
    img = img.convert("RGBA")
    datas = img.getdata()
    
    # Get background color from top-left pixel
    bg_color = datas[0]
    br, bg, bb, ba = bg_color
    
    newData = []
    for item in datas:
        r, g, b, a = item
        
        # Calculate distance
        dist = ((r - br)**2 + (g - bg)**2 + (b - bb)**2)**0.5
        
        if dist < tolerance:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
            
    img.putdata(newData)
    return img

def split_image(img, output_dir):
    width, height = img.size
    
    # Simple heuristic: Split into 3 equal parts if we can't do smart splitting easily without numpy
    # But let's try a smarter scan for empty columns
    
    # Convert to grayscale for scan
    gray = img.convert("L")
    pixels = gray.load()
    alpha = img.split()[-1].load() # Access alpha channel
    
    content_columns = []
    
    for x in range(width):
        has_content = False
        for y in range(height):
            if alpha[x, y] > 0: # Non-transparent
                has_content = True
                break
        content_columns.append(has_content)
        
    # Find segments
    segments = []
    in_segment = False
    start = 0
    for x, has_content in enumerate(content_columns):
        if has_content and not in_segment:
            start = x
            in_segment = True
        elif not has_content and in_segment:
            if x - start > 10: # Ignore noise/tiny gaps
                segments.append((start, x))
            in_segment = False
            
    if in_segment:
        segments.append((start, width))
        
    print(f"Found {len(segments)} segments.")
    
    # Names
    names = ["packet_tomato.png", "packet_potato.png", "packet_onion.png"] # Order from prompt: Tomato, Potato, Onion
    
    # Fallback to equal thirds if not 3 segments found
    if len(segments) != 3:
        print("Fallback: Splitting into equal thirds.")
        seg_w = width // 3
        segments = [(0, seg_w), (seg_w, seg_w*2), (seg_w*2, width)]
        
    # Crop and Save
    saved_files = []
    for i, (start, end) in enumerate(segments):
        if i >= len(names): break
        
        # Add some padding/margin to crop
        crop_img = img.crop((start, 0, end, height))
        
        # Trim transparent borders
        bbox = crop_img.getbbox()
        if bbox:
            crop_img = crop_img.crop(bbox)
            
        save_path = os.path.join(output_dir, names[i])
        crop_img.save(save_path)
        print(f"Saved {names[i]} to {save_path}")
        saved_files.append(names[i])
        
    return saved_files

if __name__ == "__main__":
    brain_dir = r"c:/Users/Sibiraj/.gemini/antigravity/brain/aedfe86d-ad04-4ff4-b750-b4a720e6668f"
    input_file = os.path.join(brain_dir, "uploaded_media_1770537554413.png")
    output_folder = r"c:/Users/Sibiraj/Desktop/Final year proj/static"
    
    if os.path.exists(input_file):
        print(f"Processing {input_file}...")
        try:
            img = Image.open(input_file)
            img_clean = remove_bg_color(img)
            split_image(img_clean, output_folder)
            print("Done.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Input file not found: {input_file}")
