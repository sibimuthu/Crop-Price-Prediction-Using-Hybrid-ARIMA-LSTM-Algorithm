import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Configuration
INPUT_DIR = r'c:\Users\Sibiraj\Desktop\Final year proj\static\sequence'
TARGET_WIDTH = 3840
TARGET_HEIGHT = 2160

def upscale_image(filename):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return
    
    path = os.path.join(INPUT_DIR, filename)
    try:
        with Image.open(path) as img:
            # Resize using Lanczos (High Quality)
            img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
            img_resized.save(path, quality=95, optimize=True)
            print(f"Upscaled: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

def main():
    files = os.listdir(INPUT_DIR)
    print(f"Found {len(files)} files. Starting 4K upscale...")
    
    # Use threading for faster processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(upscale_image, files)
        
    print("Upscaling complete.")

if __name__ == "__main__":
    main()
