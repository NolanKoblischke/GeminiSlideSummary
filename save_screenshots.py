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

def capture_screenshot(video, timestamp_seconds, timestamp_str, output_path):
    """Capture a screenshot at the given timestamp and save it to the output path."""
    timestamp_to_capture = timestamp_seconds - 3 if timestamp_seconds > 3 else 0
    screenshot = video.get_frame(timestamp_to_capture)
    screenshot_image = Image.fromarray(np.uint8(screenshot))
    screenshot_filename = f"slide_{timestamp_str.replace(':', '_')}.png"
    screenshot_path = os.path.join(output_path, screenshot_filename)
    screenshot_image.save(screenshot_path)
    return screenshot_filename

def generate_html(video_id, screenshots):
    """Generate an HTML file with screenshots and timestamps."""
    html_template = Template("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video {{ video_id }} Screenshots</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { text-align: center; }
            .screenshot { margin-bottom: 20px; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Screenshots for Video {{ video_id }}</h1>
        {% for screenshot in screenshots %}
        <div class="screenshot">
            <h2>Slide from 3 seconds before the transition at: {{ screenshot.timestamp }}</h2>
            <img src="{{ screenshot.filename }}" alt="Screenshot at {{ screenshot.timestamp }}">
        </div>
        {% endfor %}
    </body>
    </html>
    """)
    
    html_content = html_template.render(video_id=video_id, screenshots=screenshots)
    html_path = os.path.join('screenshots', video_id, f"{video_id}_screenshots.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Generated HTML file: {html_path}")

def process_video(video_id, timestamps):
    """Process a single video: download, capture screenshots, generate HTML, and delete the video."""
    # Create directory for screenshots
    output_dir = os.path.join('screenshots', video_id)
    os.makedirs(output_dir, exist_ok=True)

    # Download video
    video_path = download_video(video_id)

    # Capture screenshots
    screenshots = []
    with VideoFileClip(video_path) as video:
        for timestamp in timestamps:
            # Convert timestamp to seconds
            minutes, seconds = map(int, timestamp.split(':'))
            timestamp_seconds = minutes * 60 + seconds
            
            screenshot_filename = capture_screenshot(video, timestamp_seconds, timestamp, output_dir)
            screenshots.append({"timestamp": timestamp, "filename": screenshot_filename})
            print(f"Captured screenshot: {screenshot_filename}")
            #if current timestamp is last timestamp, capture screenshot at the end of the video
            if timestamp == timestamps[-1]:
                minutes, seconds = map(int, timestamp.split(':'))
                timestamp_seconds = minutes * 60 + seconds + 20 # Add 20 seconds to the last timestamp
                
                screenshot_filename = capture_screenshot(video, timestamp_seconds, timestamp, output_dir)
                screenshots.append({"timestamp": timestamp, "filename": screenshot_filename})
                print(f"Captured screenshot: {screenshot_filename}")

    # Generate HTML
    generate_html(video_id, screenshots)

    # Delete video file
    os.remove(video_path)
    print(f"Deleted video file: {video_path}")

def main(json_path):
    """Main function to process all videos in the JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    for item in data:
        video_id = item['video_id']
        timestamps = item['timestamps']
        print(f"Processing video: {video_id}")
        process_video(video_id, timestamps)
        print(f"Finished processing video: {video_id}")
        print("-" * 40)

if __name__ == "__main__":
    json_path = '28_07_2024_23:30/predicted_timestamps.json'  # Path to your JSON file
    main(json_path)