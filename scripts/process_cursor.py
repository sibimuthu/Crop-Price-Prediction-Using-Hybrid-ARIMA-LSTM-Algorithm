from PIL import Image
import os

def process_cursor(input_path, output_path):
    print(f"Processing {input_path}...")
    try:
        img = Image.open(input_path).convert("RGBA")
        
        # 1. Remove Background (Simple Color Keying)
        # Assuming the background is the top-left pixel color
        datas = img.getdata()
        bg_color = datas[0]
        threshold = 50 # Increased threshold to strip artifacts better
        
        newData = []
        for item in datas:
            # Euclidean distance
            dist = ((item[0] - bg_color[0])**2 + (item[1] - bg_color[1])**2 + (item[2] - bg_color[2])**2)**0.5
            if dist < threshold:
                newData.append((255, 255, 255, 0)) # Transparent
            else:
                newData.append(item)
        
        img.putdata(newData)
        
        # 2. Trim empty space
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        # 3. Resize to 25px
        img.thumbnail((35, 35), Image.Resampling.LANCZOS)
        
        # 4. Save
        img.save(output_path)
        print(f"Saved cursor to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    brain_dir = r"c:/Users/Sibiraj/.gemini/antigravity/brain/aedfe86d-ad04-4ff4-b750-b4a720e6668f"
    input_file = os.path.join(brain_dir, "uploaded_media_1770551461117.png")
    output_file = r"c:/Users/Sibiraj/Desktop/Final year proj/static/cursor.png"
    
    if os.path.exists(input_file):
        process_cursor(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")
