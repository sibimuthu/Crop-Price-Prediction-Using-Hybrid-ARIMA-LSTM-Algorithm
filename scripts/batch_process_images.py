from PIL import Image
import os

def remove_bg_color(img_path, output_path, tolerance=40):
    try:
        print(f"Processing {os.path.basename(img_path)}...")
        img = Image.open(img_path).convert("RGBA")
        datas = img.getdata()
        
        # Get background color from top-left pixel
        bg_color = datas[0]
        
        newData = []
        for item in datas:
            # Euclidean distance
            dist = ((item[0] - bg_color[0])**2 + (item[1] - bg_color[1])**2 + (item[2] - bg_color[2])**2)**0.5
            if dist < tolerance:
                newData.append((255, 255, 255, 0)) # Transparent
            else:
                newData.append(item)
        
        img.putdata(newData)
        
        # Trim empty space
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
            
        img.save(output_path)
        print(f"Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

if __name__ == "__main__":
    brain_dir = r"c:/Users/Sibiraj/.gemini/antigravity/brain/aedfe86d-ad04-4ff4-b750-b4a720e6668f"
    output_dir = r"c:/Users/Sibiraj/Desktop/Final year proj/static"
    
    files = [
        "tomato_bengaluru_realistic_1770553985455.png",
        "tomato_roma_realistic_1770554023375.png",
        "tomato_cherry_realistic_1770554041227.png",
        "onion_big_red_realistic_1770554215229.png",
        "onion_small_shallot_realistic_1770554246994.png",
        "onion_white_realistic_1770554312898.png",
        "onion_spring_realistic_1770554331980.png",
        "potato_large_realistic_1770554347578.png",
        "potato_baby_realistic_1770554400790.png",
        "tomato_beefsteak_realistic_1770554417350.png",
        "tomato_yellow_realistic_1770554433523.png"
    ]
    
    for f in files:
        input_path = os.path.join(brain_dir, f)
        output_path = os.path.join(output_dir, f)
        
        if os.path.exists(input_path):
            remove_bg_color(input_path, output_path)
        else:
            print(f"File not found: {input_path}")
