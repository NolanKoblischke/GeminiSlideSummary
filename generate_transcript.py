from pytubefix import YouTube
import google.generativeai as genai
import os
import time
import json
from moviepy.editor import VideoFileClip
from tqdm import tqdm
def clean_json(text):
    text = text.replace("\n", "")
    text = text.replace("\t", "")
    text = text.replace('```json', '')
    text = text.replace('```', '')
    return text
def seconds_to_mmss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def generate_transcript(video_path, youtube_id):
    # Initialize the genai client with your API key
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    start_time_s = video_path.split('_')[1]
    end_time_s = video_path.split('_')[2]
    start_time = seconds_to_mmss(int(start_time_s))
    end_time = seconds_to_mmss(int(end_time_s.split('.')[0]))
    start_time_example = seconds_to_mmss(int(start_time_s))
    start_time_example2 = seconds_to_mmss(int(start_time_s)+30)

    transcript_prompt_template = f"""
    You are an AI assistant tasked with generating a transcript for an academic lecture video. Your goal is to accurately capture the spoken content and match it with the visual slides.

    Instructions:
    1. Generate a transcript of the entire lecture.
    2. Ensure that the transcript captures all spoken content, including technical terms and explanations.
    3. Include timestamps of each sentence.
    4. Timestamps should be exactly every 30 seconds.
    5. Output in JSON format with the structure of the following example:
    {{
        "transcript": [
            {{
                "timestamp": "{start_time_example}",
                "text": "Hello, today I will be introducing how to measure the TRGB using APOGEE stellar spectra... lorem ipsum..."
            }},
            {{
                "timestamp": "{start_time_example2}",
                "text": "The TRGB is a key distance indicator in astronomy, and its measurement is crucial for understanding the scale of the universe... lorem ipsum..."
            }}
        ]
    }}

    Your task is to generate a complete and accurate transcript for the given video.
    All timestamps should be inbetween {start_time} and {end_time} which are the start and end times of this particular video segment.
    """

    video_file = genai.upload_file(path=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
        print(video_file.state.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)

    file = genai.get_file(name=video_file.name)
    print(f"Retrieved file '{file.display_name}' as: {video_file.uri}")

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    generation_config = genai.GenerationConfig(temperature=1.0)
    tries = 0
    n_max_tries = 10
    print('Generating transcript...')
    while tries < n_max_tries:
        try:
            response = model.generate_content([transcript_prompt_template, file], generation_config=generation_config)
            transcript = response.text
            break
        except Exception as e:
            tries += 1
            print(f"Attempt {tries} failed with error: {e}")
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print("Resource exhausted error encountered. Sleeping for 60 seconds before retrying.")
                time.sleep(60)
            if tries >= n_max_tries:
                print("Max number of attempts reached. Exiting.")
                return None
    genai.delete_file(file.name)

    #delete triple backticks and \n
    transcript = clean_json(transcript)
    #print transcript to temp file for debugging
    with open('transcript_temp.txt', 'w') as f:
        f.write(transcript)
    transcript_json = json.loads(transcript)
    return transcript_json['transcript']