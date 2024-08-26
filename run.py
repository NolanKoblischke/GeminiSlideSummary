from generate_transcript import generate_transcript
from slide_transition_agent import generate_timestamps
from save_video import save_video, delete_video
import tqdm
import os
import json
import time
import traceback
from save_screenshots import save_screenshots
from generate_paper import generate_paper

def get_seconds(timestamp):
    minutes, seconds = timestamp.split(':')
    return int(minutes)*60 + int(seconds)

filename = "video_urls.txt"
with open(filename) as file:
    youtube_urls = file.readlines()
output_dir = time.strftime("%d_%m_%Y_%H_%M")
os.makedirs(output_dir)
transcripts = []
predicted_timestamps = []
timestamp_agent_traces = []

def print_and_log(info):
    print(info)
    with open(output_dir+'/log.txt', 'a') as f:
        current_time = time.strftime("%H:%M:%S")
        f.write(f"{current_time}\n{info}\n")
start_time = time.time()
for youtube_url in tqdm.tqdm(youtube_urls, desc="Videos", unit="video"):
    try:
        print_and_log(f"Processing video: {youtube_url}")
        youtube_id = youtube_url.split('=')[1].strip()
        flash_calls = 0
        pro_calls = 0
        # Download video, add timestamp overlay on video, save video in 7-minute segments
        full_vid_path, segment_dir = save_video(youtube_id)
        print_and_log(f"Saved video for video: {youtube_id}")
        # Feed entire video to Gemini 1.5 Flash to generate full transcript with timestamps
        segment_jsons = []
        for segment in tqdm.tqdm(os.listdir(segment_dir), desc="Segments", unit="segment"):
            segment_path = os.path.join(segment_dir, segment)
            transcript = generate_transcript(segment_path, youtube_id)
            flash_calls += 1
            segment_jsons.extend(transcript)
            print_and_log(f"Saved transcript for segment: {segment}")
        try:
            segment_jsons.sort(key=lambda x: get_seconds(x['timestamp']))
        except Exception as e:
            print_and_log(f"Error sorting segments by timestamp: {e}")
        transcript_json = {
            'video_id': youtube_id,
            'transcript': segment_jsons
        }
        transcripts.append(transcript_json)
        with open(output_dir+'/transcripts.json', 'w') as f:
            json.dump(transcripts, f, indent=4)
        print_and_log(f"Saved transcript for video: {youtube_id}")

        # Feed video segments to Gemini 1.5 Pro to generate slide transition timestamps
        # Include it's verbose output in the timestamp_agent_trace for insights into what the slides contain
        entire_video_predicted_timestamps_json, timestamp_agent_trace_json, flash_calls, pro_calls = generate_timestamps(youtube_id,segment_dir, flash_calls, pro_calls)
        predicted_timestamps.append(entire_video_predicted_timestamps_json)
        timestamp_agent_traces.append(timestamp_agent_trace_json)
        with open(output_dir+'/predicted_timestamps.json', 'w') as f:
            json.dump(predicted_timestamps, f, indent=4)
        with open(output_dir+'/timestamp_agent_traces.json', 'w') as f:
            json.dump(timestamp_agent_traces, f, indent=4)
        print_and_log(f"Saved timestamps and timestamp agent trace for video: {youtube_id}")

        # Delete the video and its segments to save space
        delete_video(youtube_id)
        print_and_log(f"Deleted video files for video: {youtube_id}")
        print_and_log("Saving screenshots")
        save_screenshots(youtube_id, entire_video_predicted_timestamps_json['timestamps'], output_dir)
        print_and_log("Generating paper")
        generate_paper(youtube_id, output_dir)
        pro_calls += 1
        #print and log the number of flash and pro calls made
        print_and_log(f"Total Flash calls: {flash_calls}")
        print_and_log(f"Total Pro calls: {pro_calls}")
        print_and_log(f"-"*40+"\n\n")
    except Exception as e:
        error_message = f"Error processing video: {youtube_url}\n{type(e).__name__}: {e}\nTraceback:\n{traceback.format_exc()}"
        print_and_log(error_message)
        print_and_log(f"-"*40+"\n\n")
        continue

if os.path.exists('transcript_temp.txt'):
    os.remove('transcript_temp.txt')
import shutil
if os.path.exists('videos'):
    shutil.rmtree('videos')
runtime_hours = (time.time() - start_time)/3600
print_and_log("All videos processed successfully in {:.2f} hours or {:.2f} hours/video".format(runtime_hours, runtime_hours/len(youtube_urls)))
