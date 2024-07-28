from pytubefix import YouTube
import os
import cv2
import numpy as np
from datetime import timedelta
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm

def add_timestamp(input_file, output_file):
    # Open the video file
    video = cv2.VideoCapture(input_file)
    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = "temp_output.mp4"
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    # Define timestamp box properties
    box_width = 225
    box_height = 75
    box_color = (0, 0, 0)  # Black
    text_color = (255, 255, 255)  # White
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Calculate current time
        current_time = timedelta(seconds=frame_count/fps)
        time_str = str(current_time).split('.')[0][-5:]  # Get MM:SS format
        # Create black box
        cv2.rectangle(frame, 
                      (width - box_width, height - box_height), 
                      (width, height), 
                      box_color, 
                      -1)
        # Add timestamp text
        text_size = cv2.getTextSize(time_str, font, font_scale, font_thickness)[0]
        text_x = width - box_width + (box_width - text_size[0]) // 2
        text_y = height - (box_height - text_size[1]) // 2
        cv2.putText(frame, time_str, (text_x, text_y), font, font_scale, text_color, font_thickness)
        # Write the frame
        out.write(frame)
        frame_count += 1
    # Release everything
    video.release()
    out.release()
    cv2.destroyAllWindows()

    # Now use MoviePy to add the audio back
    video_clip = VideoFileClip(temp_output)
    original_audio = AudioFileClip(input_file)
    final_clip = video_clip.set_audio(original_audio)
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

    # Close the clips
    video_clip.close()
    original_audio.close()
    final_clip.close()

    # Optionally, remove the temporary file
    import os
    os.remove(temp_output)

def sanitize_json(text):
    # Replace single backslashes with double backslashes
    return text.replace('\\', '\\\\')

def seconds_to_mmss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def mmss_to_seconds(mmss):
    minutes, seconds = map(int, mmss.split(':'))
    return minutes * 60 + seconds

def timestamp_to_filename(timestamp):
    return timestamp.replace(':', '_')

def save_segment(start, end, video_path):
    video_dir = os.path.dirname(video_path)
    os.makedirs(f"{video_dir}/segments", exist_ok=True)
    segment_path = f"{video_dir}/segments/segment_{start}_{end}.mp4"
    ffmpeg_extract_subclip(video_path, start, end, targetname=segment_path)

def save_video(youtube_id):
    base_dir = 'videos'
    os.makedirs(base_dir+'/'+youtube_id, exist_ok=True)
    yt = YouTube('https://www.youtube.com/watch?v='+youtube_id)
    if yt.streams.filter(progressive=True, res='480p', file_extension='mp4'):
        stream = yt.streams.filter(progressive=True, res='480p', file_extension='mp4').first()
    elif yt.streams.filter(progressive=True, res='360p', file_extension='mp4'):
        stream = yt.streams.filter(progressive=True, res='360p', file_extension='mp4').first()
    else:
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').asc().first()

    original_video_path = f'{base_dir}/{youtube_id}/{youtube_id}_original.mp4'
    timestamped_video_path = f'{base_dir}/{youtube_id}/{youtube_id}.mp4'
    
    stream.download(filename=original_video_path)
    print(f"Downloaded video to {original_video_path}")
    
    # Add timestamp to the video
    add_timestamp(original_video_path, timestamped_video_path)
    print(f"Added timestamp overlay to {timestamped_video_path}")
    
    # Remove the original video to save space
    os.remove(original_video_path)
    
    # Process the video in 7-minute segments
    segment_duration = 7 * 60  # 7 minutes in seconds
    video = VideoFileClip(timestamped_video_path)
    video_length_seconds = int(video.duration)
    for start in tqdm(range(0, video_length_seconds, segment_duration)):
        end = min(start + segment_duration, video_length_seconds)
        save_segment(start, end, timestamped_video_path)
    
    return timestamped_video_path, f"{base_dir}/{youtube_id}/segments"

def delete_video(youtube_id):
    base_dir = 'videos/'
    # Delete the video and its segments
    video_dir = f"{base_dir}/{youtube_id}"
    os.system(f"rm -rf {video_dir}")
