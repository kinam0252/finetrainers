import os
import subprocess

def extract_evenly_spaced_frames_ffmpeg(video_path, output_frame_dir, total_frames=13):
    os.makedirs(output_frame_dir, exist_ok=True)
    
    # 전체 프레임 수 파악
    cmd_total = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1", video_path
    ]
    total = int(subprocess.check_output(cmd_total).decode().strip())
    
    # 균일한 인덱스 추출 (0, ..., total-1)
    indices = [0] + [int((i+1)*(total-1)/(total_frames-1)) for i in range(total_frames-2)] + [total-1]
    
    # 각 프레임 추출
    for idx, frame_number in enumerate(indices):
        output_file = os.path.join(output_frame_dir, f"frame_{idx:02d}.png")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-vf", f"select='eq(n\,{frame_number})'",
            "-vsync", "0", output_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_video_from_frames(frame_dir, output_video_path, resolution="480x480", fps=10):
    # 마지막 프레임 복사해서 49개로 확장
    last_frame = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])[-1]
    for i in range(13, 49):
        dst = os.path.join(frame_dir, f"frame_{i:02d}.png")
        src = os.path.join(frame_dir, last_frame)
        subprocess.run(["cp", src, dst])

    # 프레임을 mp4로 변환
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(frame_dir, "frame_%02d.png"),
        "-s", resolution, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_video_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_all_videos_ffmpeg(plain_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted([
        f for f in os.listdir(plain_dir) if f.lower().endswith((".mp4", ".mov", ".avi"))
    ])
    
    for idx, video_file in enumerate(video_files, start=1):
        video_path = os.path.join(plain_dir, video_file)
        temp_frame_dir = f"temp_frames_{idx}"
        
        extract_evenly_spaced_frames_ffmpeg(video_path, temp_frame_dir)
        output_path = os.path.join(output_dir, f"{idx}.mp4")
        make_video_from_frames(temp_frame_dir, output_path)

        # 정리
        subprocess.run(["rm", "-rf", temp_frame_dir])
        print(f"Saved: {output_path}")

# 예시 사용법:
process_all_videos_ffmpeg("plain/", "videos/")
