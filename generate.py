from pytube import YouTube
import google.generativeai as genai
import os
import time
import json
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
import pandas as pd

filename = "video_urls.txt"
with open(filename) as file:
    youtube_urls = file.readlines()

for youtube_url in youtube_urls:
    print(youtube_url)

    youtube_id = youtube_url.split('=')[1]

    #check if directory called youtube_id exists
    if not os.path.exists(youtube_id):
        os.makedirs(youtube_id, exist_ok=True)
    else:
        print(f"Directory already exists for {youtube_id}, please delete it and try again.")
        continue

    yt = YouTube(youtube_url)
    if yt.streams.filter(res='480p', file_extension='mp4'):
        stream = yt.streams.filter(res='480p', file_extension='mp4').first()
    elif yt.streams.filter(res='360p', file_extension='mp4'):
        stream = yt.streams.filter(res='360p', file_extension='mp4').first()
    else:
        stream = yt.streams.filter(res='480p', file_extension='mp4').order_by('resolution').asc().first()
    video_path = f'{youtube_id}/{youtube_id}.mp4'
    stream.download(filename=video_path)
    print(f"Downloaded video to {video_path}")



    def sanitize_json(text):
        # Replace single backslashes with double backslashes
        text = text.replace('\\', '\\\\')
        return text

    # Initialize the genai client with your API key
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    # Define the prompt templates
    timestamp_prompt_template = """You are an AI assistant tasked with identifying the major slide changes in a segment of an academic lecture video. Your goal is to match the exact timing of true slide changes with high accuracy. Follow these instructions meticulously:

    1. Critical instruction: Predict around 5 timestamps in total for this segment.

    2. Time format:
    - Use "MM:SS" format strictly

    3. Lecture structure:
    - This segment starts at "{start_time}".
    - This segment ends at "{end_time}".

    4. Timing rules:
    - Timestamps should be 2-3 minutes apart.
    - No two timestamps should be less than 30 seconds apart.

    5. Slide change identification:
    - Include all major topic transitions and significant content changes.
    - Pay close attention to the pacing in different parts of the lecture.

    6. Precision:
    - Do not artificially round timestamps; use the exact time you observe a change. Do NOT round to the nearest 00 or 10 or 15 seconds.

    Output in this JSON format. For example:
    {{
    "slides": [
        {{
        "timestamp": "{start_time_example}",
        }},
        ...
        {{
        "timestamp": "{end_time_example}",
        }}
    ]
    }}
    Note: The timestamps above are examples and not related to the video. You should provide timestamps based on the actual slide changes.

    Critical reminders:
    - You must output around 5 timestamps.
    - Be precise with your timing. Don't round or approximate.
    - Include all significant content changes.

    Before finalizing, verify:
    1. You have around 5 timestamps.
    2. Timestamps are distributed according to the guidelines provided.
    3. No timestamps are less than 30 seconds apart.

    Your primary goal is to accurately capture all major slide changes throughout the entire segment.
    """

    description_prompt_template = """You are an AI assistant tasked with describing the content of slides in a segment of an academic lecture video. Here are the slides captured from the segment starting at "{start_time}" and ending at "{end_time}". Please provide a detailed description for each slide.
    - For each slide, you should provide a brief summary of the main points the speaker talks about and how they relate to the figures and text on the slide.
    - You are producing this for an audience that has a strong PhD-level understanding of astrophysics, so you should use technical language and assume that the audience is familiar with the concepts being discussed.
    - Your audience has already seen the talk up to this point, so you need to focus only on these slides.
    - Be incredibly technical. Feel free to use LaTeX for equations and jargon.
    - Each summary should only be three sentences long.
    - The first sentence should summarize the visuals and text of the slide itself.
    - The last two sentences should summarize the speaker's discussion of the slide.
    - You will be provided the start times of each slide.
    - You will be provided an OCR of the slide as well, use it to identify what slide is being discussed.
    - Do not trust the OCR. The OCR is likely entirely wrong and full of errors. Only trust the first words in the OCR since that is likely the title of the slide. Trust the visuals and audio over the OCR everytime.
    - The summaries should cover the 3-5 minutes of discussion that follows the start time you are given.
    - Do NOT include double quotes in the descriptions. Escaping double quotes inside the description will not work. Only use single quotes if needed.
    - If you suspect two slides to be duplicates, you can skip one of them by not outputing a JSON object for it. Only provide descriptions for unique slides. 

    Output in this JSON format:
    {{
    "slides": [
        {{
        "timestamp": "MM:SS",
        "description": "Your three-sentence description here.",
        }},
        ...
    ]
    }}
    Here is an example of the expected output for an entirely unrelated lecture:
    {{
    "slides": [
        {{
        "timestamp": "{timestamp_1}",
        "description": "The slide shows the Hertzsprung-Russell diagram with main sequence, red giants, and white dwarfs and text talking about stellar evolution. The speaker discusses key factors influencing stellar evolution include: 1) More massive stars evolve faster, are hotter, more luminous, and more likely to undergo supernovae. 2) Stars with higher metallicity have higher opacity and experience greater mass loss through stellar winds.",
        }},
        ...
        {{
        "timestamp": "{timestamp_N}",
            "description": "The slide includes a schematic diagram of the magnetorotational instability (MRI) in accretion disks from Adams+2007, with arrows indicating magnetic field lines and velocity vectors. The speaker explains how differential rotation in the disk stretches the magnetic field lines, leading to amplification of the initial magnetic field and enhanced angular momentum transport; this process is crucial for explaining the observed accretion rates in astrophysical systems ranging from protoplanetary disks to active galactic nuclei, as it provides a mechanism for efficient outward transport of angular momentum and inward flow of matter.",
        }}
    ]
    }}
    ---
    **IMPORTANT**: Do not copy or use the above example descriptions provided. The above descriptions about HR diagrams and MRI in accretion disks were from entirely different lectures and are only provided as an example. Do NOT copy them, rather use them as a example for the expected style. Generate unique, relevant descriptions for each slide based on the given information.

    Your primary goal is to provide a detailed and accurate description for the following slides. Here are the start times: {combined_timestamp_ocr}.
    """

    summary_prompt = """# Objective: Summarize an entire astrophysics lecture in the style of a Nature-level summary.

    ### Instructions:

    1. **Content Analysis**:
    - Analyze the lecture, including visual (slides, diagrams, equations) and audio content.

    2. **Summary Structure**:
    - Write 3-5 paragraphs, each covering a major theme or section.
    - Use a formal, academic tone.

    3. **Technical Proficiency**:
    - Target PhD-level astrophysicists; use advanced technical language and jargon.
    - Incorporate LaTeX equations, e.g., $H^2 = \\frac{8\\pi G}{3}\\rho - \\frac{kc^2}{a^2} + \\frac{\\Lambda c^2}{3}$.

    4. **Content Inclusion**:
    - State hypotheses, objectives, and key frameworks.
    - Detail significant data, results, methodologies, and diagrams.
    - Connect to broader cosmological paradigms.

    5. **Critical Analysis**:
    - Identify the scientific narrative, contentious points, and areas of uncertainty.
    - Assess the lecture's contribution to astrophysics.

    6. **Scientific Precision**:
    - Ensure accuracy in terminology, measurements, and equations.

    7. **Format**:
    - Start with a concise introduction of the thesis and speaker.
    - Conclude with the lecture's implications for astrophysics.
    - Include paragraph breaks for ease of reading."""

    def create_summary(video_path):
        # Initialize start and end times for the entire video
        # Upload the video segment
        
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
        print('Generating summary...')
        while tries < n_max_tries:
            try:
                response = model.generate_content([summary_prompt, file], request_options={"timeout": 600}, generation_config=generation_config)
                summary = response.text
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
        return summary

    # Function to convert seconds to MM:SS format
    def seconds_to_mmss(seconds):
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    # Function to convert MM:SS format to seconds
    def mmss_to_seconds(mmss):
        minutes, seconds = map(int, mmss.split(':'))
        return minutes * 60 + seconds

    # Function to capture screenshot at a specific timestamp
    def capture_screenshot(video, timestamp, output_path):
        screenshot = video.get_frame(timestamp)
        screenshot_image = Image.fromarray(np.uint8(screenshot))
        screenshot_path = os.path.join(output_path, f"slide_{timestamp}.png")
        screenshot_image.save(screenshot_path)
        return screenshot_path

    # Function to convert timestamp to a valid filename
    def timestamp_to_filename(timestamp):
        return timestamp.replace(':', '_')

    # Function to process a single segment
    def process_segment(start, end, video_path, timestamp_prompt_template, description_prompt_template, OCRReader):
        start_time = seconds_to_mmss(start)
        end_time = seconds_to_mmss(end)
        start_time_example = seconds_to_mmss(start + 17)
        end_time_example = seconds_to_mmss(end - 37)
        
        # Create a temporary video segment
        # Check if {youtube_id}/segments directory exists
        if not os.path.exists(f"{youtube_id}/segments"):
            os.makedirs(f"{youtube_id}/segments", exist_ok=True)
        segment_path = f"{youtube_id}/segments/segment_{start}_{end}.mp4"
        ffmpeg_extract_subclip(video_path, start, end, targetname=segment_path)
        
        # Upload the video segment
        video_file = genai.upload_file(path=segment_path)
                
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
            print(video_file.state.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        file = genai.get_file(name=video_file.name)
        print(f"Retrieved file '{file.display_name}' as: {video_file.uri}")
        
        # Create the prompt for the current segment to get timestamps
        timestamp_prompt = timestamp_prompt_template.format(start_time=start_time, end_time=end_time, start_time_example=start_time_example, end_time_example=end_time_example)
        
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        generation_config = genai.GenerationConfig(temperature=0.0, response_mime_type="application/json")
        tries = 0
        n_max_tries = 10
        print('Making timestamps...')
        while tries < n_max_tries:
            try:
                if tries > 0:
                    print("Timestamp Retrying...")
                response = model.generate_content([timestamp_prompt, file], request_options={"timeout": 600}, generation_config=generation_config)
                data = json.loads(sanitize_json(response.text))
                break
            except Exception as e:
                tries += 1
                print(f"Timestamp Attempt {tries} failed with error: {e}")
                if "Resource has been exhausted" in str(e) or "429" in str(e):
                    print("Resource exhausted error encountered. Sleeping for 60 seconds before retrying.")
                    time.sleep(60)
                if tries >= n_max_tries:
                    print("Max number of attempts reached. Exiting.")
                    return None, None, None
        # Parse the response and get the timestamps
        timestamps = [mmss_to_seconds(t['timestamp']) for t in data['slides']]
        
        # Capture screenshots at the predicted timestamps
        screenshots = []
        screenshot_ocrs = []
        video = VideoFileClip(video_path)
        output_path = f"{youtube_id}/slides"
        os.makedirs(output_path, exist_ok=True)
        for timestamp in timestamps:
            # Capture a screenshot 5 seconds after the predicted timestamp
            screenshot_path = capture_screenshot(video, timestamp+5, output_path)
            screenshot_ocr = OCRReader.readtext(screenshot_path)
            filtered_ocr = ' '.join([item[1] for item in screenshot_ocr if len(item[1]) >= 5])
            screenshots.append(screenshot_path)
            screenshot_ocrs.append(filtered_ocr)
        
        timestamp_list = [seconds_to_mmss(t) for t in timestamps]
        combined_timestamp_ocr = ""
        for i, (timestamp, ocr) in enumerate(zip(timestamp_list, screenshot_ocrs)):
            combined_timestamp_ocr += f"Slide {i+1}:\nTimestamp: {timestamp}\nOCR of Slide: {ocr}\n\n"
        description_prompt = description_prompt_template.format(start_time=start_time, end_time=end_time, timestamp_1=start_time_example, timestamp_N=end_time_example, combined_timestamp_ocr=combined_timestamp_ocr)
        # Get descriptions for the slides
        print('Making description...')
        print(timestamp_list)
        generation_config = genai.GenerationConfig(temperature=1.0, response_mime_type="application/json")
        tries = 0
        response_text = None
        while tries < n_max_tries:
            try:
                if tries > 0:
                    print("Description Retrying...")
                response = model.generate_content([description_prompt, file], request_options={"timeout": 600}, generation_config=generation_config)
                response_text = response.text
                descriptions_data = json.loads(sanitize_json(response.text))
                break
            except Exception as e:
                tries += 1
                print(f"Description Attempt {tries} failed with error: {e}")
                print(response_text)
                if "Resource has been exhausted" in str(e) or "429" in str(e):
                    print("Resource exhausted error encountered. Sleeping for 60 seconds before retrying.")
                    time.sleep(60)
                if tries >= n_max_tries:
                    print("Max number of attempts reached. Exiting.")
                    return None, None, None
        genai.delete_file(file.name)
        descriptions = [t['description'] for t in descriptions_data['slides']]
        timestamps = [t['timestamp'] for t in descriptions_data['slides']]

        # Remove screenshots that are not in timestamps
        timestamps = [int(mmss_to_seconds(t)) for t in timestamps]
        timestamps_plus_5 = [t+5 for t in timestamps]
        screenshots_clean = []
        descriptions_clean = []
        timestamps_clean = []
        screenshots_removed = []
        for screenshot in screenshots:
            timestamp = int(screenshot.split('_')[-1].split('.')[0])
            if timestamp in timestamps_plus_5:
                screenshots_clean.append(screenshot)
                index = timestamps_plus_5.index(timestamp)
                descriptions_clean.append(descriptions[index])
                timestamps_clean.append(timestamps[index])
            else:
                screenshots_removed.append(screenshot)
        if len(screenshots_removed) > 0:
            print(f'Number of screenshots without descriptions: {len(screenshots_removed)}')
            print('Screenshots removed since no description generated:', screenshots_removed)

        return timestamps_clean, descriptions_clean, screenshots_clean

    # Process the video in 10-minute segments
    segment_duration = 10 * 60  # 10 minutes in seconds
    video = VideoFileClip(video_path)
    video_length_seconds = int(video.duration)

    predicted_descriptions = []
    predicted_timestamps = []
    predicted_screenshots = []

    OCRReader = easyocr.Reader(['en'])
    # Use ThreadPoolExecutor to process segments in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        # Skip last segment since that is usually the question period
        for start in range(0, video_length_seconds - segment_duration, segment_duration):
            end = min(start + segment_duration, video_length_seconds)
            futures.append(executor.submit(process_segment, start, end, video_path, timestamp_prompt_template, description_prompt_template, OCRReader))

        for future in tqdm(as_completed(futures), total=len(futures)):
            timestamps, descriptions, screenshots = future.result()
            predicted_timestamps.extend(timestamps)
            predicted_descriptions.extend(descriptions)
            predicted_screenshots.extend(screenshots)

    predicted_summary = create_summary(video_path)
    
    #save predicted_summary to a file
    with open(f'{youtube_id}/entire_video_summary_{youtube_id}.txt', 'w') as file:
        file.write(predicted_summary)

    print(len(predicted_timestamps), len(predicted_descriptions), len(predicted_screenshots))
    def sort_predictions(timestamps, descriptions, screenshots):
        #get None indices in either timestamps or descriptions or screenshots
        not_none_indices = [i for i, (t, d, s) in enumerate(zip(timestamps, descriptions, screenshots)) if t is not None or d is not None or s is not None]
        timestamps = [timestamps[i] for i in not_none_indices]
        descriptions = [descriptions[i] for i in not_none_indices]
        screenshots = [screenshots[i] for i in not_none_indices]
        print(len(timestamps), len(descriptions), len(screenshots))
        # Get sorted indices based on timestamps
        # sorted_indices = sorted(range(len(timestamps)), key=lambda i: convert_to_seconds(timestamps[i]))
        sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])

        # Sort the lists according to the sorted indices
        sorted_timestamps = [timestamps[i] for i in sorted_indices]
        sorted_descriptions = [descriptions[i] for i in sorted_indices]
        sorted_screenshots = [screenshots[i] for i in sorted_indices]

        return sorted_timestamps, sorted_descriptions, sorted_screenshots

    predicted_timestamps, predicted_descriptions, predicted_screenshots = sort_predictions(predicted_timestamps, predicted_descriptions, predicted_screenshots)

    #save all to a csv
    data = {'timestamp': predicted_timestamps, 'description': predicted_descriptions, 'screenshot_path': predicted_screenshots}
    df = pd.DataFrame(data)
    df.to_csv(f'{youtube_id}/predictions_{youtube_id}.csv', index=False)


    # Define the HTML template
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Lecture Slides and Descriptions</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f0f0;
            }}
            .container {{
                width: 95%;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #ffffff;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            .slide {{
                display: flex;
                flex-direction: column;
                margin-bottom: 20px;
                border-bottom: 1px solid #dddddd;
                padding-bottom: 10px;
            }}
            .slide img {{
                width: 100%;
                max-width: 800px;
                height: auto;
                margin-bottom: 10px;
                border: 1px solid #dddddd;
                border-radius: 5px;
            }}
            .description {{
                width: 100%;
            }}
            h1 {{
                text-align: center;
                color: #333333;
            }}
            .youtube-link {{
                text-align: center;
                margin-bottom: 20px;
            }}
            h2 {{
                font-size: 1.2em;
                color: #333333;
                margin: 0 0 10px 0;
            }}
            p {{
                margin: 0;
                color: #555555;
                font-size: 1em;
            }}
            .summary {{
                margin: 20px auto;
                max-width: 100%;
                text-align: left;
                font-size: 0.9em;
            }}
            
            @media (min-width: 768px) {{
                .slide {{
                    flex-direction: row;
                    align-items: flex-start;
                }}
                .slide img {{
                    width: 60%;
                    margin-right: 20px;
                    margin-bottom: 0;
                }}
                .description {{
                    width: 40%;
                }}
                p {{
                    font-size: 1.2em;
                }}
            }}

            @media print {{
                body {{
                    background-color: #ffffff;
                }}
                .container {{
                    width: 100%;
                    max-width: none;
                    margin: 0;
                    padding: 0;
                    box-shadow: none;
                }}
                .slide {{
                    page-break-inside: avoid;
                    display: flex;
                    flex-direction: row;
                    align-items: flex-start;
                    margin-bottom: 20px;
                }}
                .slide img {{
                    width: 50%;
                    max-width: 50%;
                    height: auto;
                    margin-right: 20px;
                }}
                .description {{
                    width: 45%;
                }}
                h2 {{
                    font-size: 14pt;
                    margin-top: 0;
                }}
                p {{
                    font-size: 12pt;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{youtube_title}</h1>
            <div class="youtube-link">
                <h2>Lecture Slides and Descriptions</h2>
                <h2><a href="{youtube_url}">Youtube Link</a></h2>
            </div>
            <div class="summary">
                <p><b>Summary: </b><br>{summary}</p>
            </div>
            <br><br>
            {content}
        </div>
    </body>
    </html>
    """

    # Generate the content for each slide
    slide_template = """
    <div class="slide">
        <img src="{image_path}" alt="Slide Image">
        <div class="description">
            <h2>Slide {slide_number}</h2>
            <p>{description}</p>
        </div>
    </div>
    """

    # Create the content by iterating through the slides and descriptions
    content = ""
    for I, row in df.iterrows():
        ts = row['timestamp']
        description = row['description']
        slide_filename = row['screenshot_path'].replace(youtube_id+'/', '')
        content += slide_template.format(
            image_path=slide_filename,
            slide_number=I + 1,
            description=description
        )

    # Create the final HTML content
    youtube_url = 'https://www.youtube.com/watch?v=' + video_path.split('.')[0]
    html_content = html_template.format(content=content, youtube_url=youtube_url, summary=predicted_summary.replace('\n', '<br>'), youtube_title=yt.title)

    # Save the HTML content to a file
    html_file = f"{youtube_id}/{youtube_id}_output.html"
    with open(html_file, mode='w') as file:
        file.write(html_content)

    print(f"HTML file saved as {html_file}.")