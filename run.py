from generate_transcript import generate_transcript
from slide_transition_agent import generate_timestamps
from save_video import save_video, delete_video
import tqdm
import os
import json

def get_seconds(timestamp):
    minutes, seconds = timestamp.split(':')
    return int(minutes)*60 + int(seconds)

filename = "video_urls.txt"
with open(filename) as file:
    youtube_urls = file.readlines()

transcripts = []
predicted_timestamps = []
timestamp_agent_traces = []
for youtube_url in tqdm.tqdm(youtube_urls, desc="Videos", unit="video"):
    
    print(f"Processing video: {youtube_url}")
    youtube_id = youtube_url.split('=')[1].strip()

    # Download video, add timestamp overlay on video, save video in 7-minute segments
    full_vid_path, segment_dir = save_video(youtube_id)

    # Feed entire video to Gemini 1.5 Flash to generate full transcript with timestamps
    segment_jsons = []
    for segment in tqdm.tqdm(os.listdir(segment_dir), desc="Segments", unit="segment"):
        segment_path = os.path.join(segment_dir, segment)
        transcript = generate_transcript(segment_path, youtube_id)
        segment_jsons.extend(transcript)
    try:
        segment_jsons.sort(key=lambda x: get_seconds(x['timestamp']))
    except Exception as e:
        print("Error sorting segments by timestamp:", e)
    transcript_json = {
        'video_id': youtube_id,
        'transcript': segment_jsons
    }
    transcripts.append(transcript_json)
    with open('transcripts.json', 'w') as f:
        json.dump(transcripts, f, indent=4)
    print('Saved transcript for video:', youtube_id)

    # Feed video segments to Gemini 1.5 Pro to generate slide transition timestamps
    # Include it's verbose output in the timestamp_agent_trace for insights into what the slides contain
    entire_video_predicted_timestamps_json, timestamp_agent_trace_json = generate_timestamps(youtube_id,segment_dir)
    predicted_timestamps.append(entire_video_predicted_timestamps_json)
    timestamp_agent_traces.append(timestamp_agent_trace_json)
    with open('predicted_timestamps.json', 'w') as f:
        json.dump(predicted_timestamps, f, indent=4)
    with open('timestamp_agent_traces.json', 'w') as f:
        json.dump(timestamp_agent_traces, f, indent=4)
    print('Saved timestamps and timestamp agent trace for video:', youtube_id)

    # Delete the video and its segments to save space
    delete_video(youtube_id)
    print('Deleted video files for video:', youtube_id)
    print('\n\n\n\n\n')

if os.path.exists('transcript_temp.txt'):
    os.remove('transcript_temp.txt')
if os.path.exists('videos'):
    os.rmdir('videos')