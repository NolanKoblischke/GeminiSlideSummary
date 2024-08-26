import json
import os
from pytubefix import YouTube
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
from jinja2 import Template

def download_video(video_id):
    """Download the video and return the path to the downloaded file."""
    url = f'https://www.youtube.com/watch?v={video_id}'
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    video_path = stream.download(filename=f'{video_id}.mp4')
    print(f"Downloaded video: {video_id}")
    return video_path

def capture_slide(video, timestamp_seconds, timestamp_str, output_path):
    """Capture a slide at the given timestamp and save it to the output path."""
    timestamp_to_capture = timestamp_seconds - 3 if timestamp_seconds > 3 else 0
    slide = video.get_frame(timestamp_to_capture)
    slide_image = Image.fromarray(np.uint8(slide))
    slide_filename = f"slide_{timestamp_str.replace(':', '_')}.png"
    slide_path = os.path.join(output_path, slide_filename)
    slide_image.save(slide_path)
    return slide_filename

def generate_html(video_id, slides, output_dir):
    """Generate an HTML file with slides and timestamps."""
    html_template = Template("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video {{ video_id }} Slides</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { text-align: center; }
            .slide { margin-bottom: 20px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Slides for Video {{ video_id }}</h1>
        {% for slide in slides %}
        <div class="slide">
            <h2>Slide from 3 seconds before the transition at: {{ slide.timestamp }}</h2>
            <img src="{{ slide.filename }}" alt="Slide at {{ slide.timestamp }}">
        </div>
        {% endfor %}
    </body>
    </html>
    """)
    
    html_content = html_template.render(video_id=video_id, slides=slides)
    html_path = os.path.join(output_dir, f"{video_id}_slides.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated HTML file: {html_path}")

def save_screenshots(video_id, timestamps, directory):
    """Process a single video: download, capture slides, generate HTML, and delete the video."""
    # Create directory for slides should be directory/slides/video_id
    output_dir = os.path.join(directory, 'slides', video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Download video
    video_path = download_video(video_id)

    # Capture slides
    slides = []
    with VideoFileClip(video_path) as video:
        for timestamp in timestamps:
            # Convert timestamp to seconds
            minutes, seconds = map(int, timestamp.split(':'))
            timestamp_seconds = minutes * 60 + seconds
            
            slide_filename = capture_slide(video, timestamp_seconds, timestamp, output_dir)
            slides.append({"timestamp": timestamp, "filename": slide_filename})
            print(f"Captured slide: {slide_filename}")
            #if current timestamp is last timestamp, capture slide at the end of the video
            if timestamp == timestamps[-1]:
                minutes, seconds = map(int, timestamp.split(':'))
                timestamp_seconds = minutes * 60 + seconds + 20 # Add 20 seconds to the last timestamp
                
                slide_filename = capture_slide(video, timestamp_seconds, timestamp, output_dir)
                slides.append({"timestamp": timestamp, "filename": slide_filename})
                print(f"Captured slide: {slide_filename}")

    # Generate HTML
    generate_html(video_id, slides, output_dir)

    # Delete video file
    os.remove(video_path)
    print(f"Deleted video file: {video_path}")
