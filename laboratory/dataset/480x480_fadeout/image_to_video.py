import os
import subprocess
import tempfile
from PIL import Image
import numpy as np

def create_video_from_image(image_path, output_path, num_frames=49, fps=10, save_sample=False):
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load and resize image
        original_image = Image.open(image_path)
        original_image = original_image.resize((480, 480), Image.LANCZOS)
        original_array = np.array(original_image)
        
        # Save frames with fadeout effect
        frames = []
        for i in range(num_frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
            
            # Calculate fadeout factor (1.0 to 0.0)
            if i < 13:
                fade_factor = 1.0 - (i / 12)  # 13번째 프레임까지 fadeout
            else:
                fade_factor = 0.0  # 13번째 이후는 완전히 검은색
            
            # Apply fadeout
            faded_array = (original_array * fade_factor).astype(np.uint8)
            faded_image = Image.fromarray(faded_array)
            faded_image.save(frame_path)
            frames.append(faded_image)
        
        # Create video from frames
        concat_cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%04d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            output_path
        ]
        subprocess.run(concat_cmd)
        
        # Save sample.png if requested
        if save_sample:
            # Create a new image with all frames concatenated horizontally
            total_width = 480 * len(frames)
            sample_image = Image.new('RGB', (total_width, 480))
            
            # Paste each frame
            for i, frame in enumerate(frames):
                sample_image.paste(frame, (i * 480, 0))
            
            # Save the sample image
            sample_path = os.path.join(os.path.dirname(output_path), 'sample.png')
            sample_image.save(sample_path)
            print(f"Created sample image: {sample_path}")

def main():
    # Input and output directories
    input_dir = '/data/kinamkim/finetrainers/laboratory/dataset/480x480/images'
    output_dir = 'videos'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files and sort them
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Process each image
    for i, image_file in enumerate(image_files, 1):
        print(f"Processing {image_file}...")
        
        # Input and output paths
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f'{i}.mp4')
        
        # Create video from image
        create_video_from_image(image_path, output_path, save_sample=(i==1))
        print(f"Created video: {output_path}")

if __name__ == "__main__":
    main() 