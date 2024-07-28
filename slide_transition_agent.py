from tqdm import tqdm
import google.generativeai as genai
import os
import time

def seconds_to_mmss(seconds):
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def generate_timestamps(youtube_id, segment_dir):

    segment_paths = os.listdir(segment_dir)
    start_timestamps = [int(filename.split('_')[1]) for filename in segment_paths]
    end_timestamps = [int(filename.split('_')[2].split('.')[0]) for filename in segment_paths]
    #sort the segments by start time
    sorted_indices = sorted(range(len(start_timestamps)), key=lambda k: start_timestamps[k])
    segment_paths = [segment_paths[i] for i in sorted_indices]
    start_timestamps = [seconds_to_mmss(start_timestamps[i]) for i in sorted_indices]
    end_timestamps = [seconds_to_mmss(end_timestamps[i]) for i in sorted_indices]
    video_files = []
    for segment_path in tqdm(segment_paths, desc="Uploading segments", unit="segment"):
        segment_path = f"{segment_dir}/{segment_path}"
        video_file = genai.upload_file(path=segment_path)

        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
            print("PROCESSING " + str(video_file.name) + " " + str(segment_path))

        if video_file.state.name == "FAILED":
            print(f"File processing failed: {video_file.name} {segment_path}")
            raise ValueError(video_file.state.name)
        video_files.append(video_file)

    def submit_slide_transitions(timestamps: list, concluding_slide: bool = False):
        """Submit slide transitions at the given timestamps.

        Args:
            timestamps: A list of timestamps of the slide transitions in the format MM:SS.
            concluding_slide: A boolean indicating if the last slide is a concluding slide.
        """
        for timestamp in timestamps:
            print(f"Slide transition submitted at {timestamp}")
        if concluding_slide:
            print("Concluding slide detected.")

    slide_transition_fn = genai.protos.FunctionDeclaration(
        name="submit_slide_transitions",
        description="Submit slide transitions at the given timestamps and indicate if the last slide is a concluding slide.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "timestamps": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                    description="A list of timestamps of the slide transitions in the format MM:SS. Can be empty if there are no slide transitions."
                ),
                "concluding_slide": genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN,
                    description="Indicates if the last slide is a concluding slide."
                ),
            },
            required=["timestamps"]
        )
    )

    initial_prompt = """System: I need you to meticulously identify ALL slide transitions in this video, even if they seem minor. A slide transition happens whenever the content on the presentation changes.
    Here's what I need you to do for EACH transition:
    1. Give me the timestamp. 
    2. Describe the visual change in a sentence or two. For example: "A new slide appears with the title 'Finding the Gaia Sausage in large, bold font at the top. Below the title is a bulleted list."
    3. If it's a minor change, like an animation, new plot, or revealing a new bullet point, then you must say "Type: MINOR CHANGE". If it's an entirely new slide then you must say "Type: NEW SLIDE" and tell me why you think it's a new slide. For example: "This is clearly a new slide because the title, layout, and content are completely different from the previous slide."
    4. Pay special attention to identify if it could be the end of the lecture and indicate with "Type: END OF LECTURE". Make a note if the speaker stops talking for an extended period since that might be the conclusion. Make a note if there is any mention of 'taking questions' which could indicate its the concluding slide. Do not submit any timestamps after the speaker has concluded the presentation and it is the question-and-answer period. 
    Be as detailed as possible! I want to be absolutely sure we don't miss any slide changes.
    Example:
    "04:30: The screen transitions to a slide with the title 'Finding the Gaia Sausage' at the top. A diagram depicting a network architecture with various boxes and arrows fills the majority of the slide. This is a new slide because the title, visual content, and layout are distinct from the previous slide about Proper Motion. Type: NEW SLIDE"
    """
    convo_history = initial_prompt
    print(convo_history)

    text_model = genai.GenerativeModel(
        model_name='gemini-1.5-pro',
        generation_config={'temperature': 0.0}
    )
    tool_model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[slide_transition_fn]
    )

    entire_video_predicted_timestamps = []
    last_message = ""

    for i, video_file in enumerate(tqdm(video_files, desc="Feeding segments to the model", unit="segment")):
        print(f"System: Here's the video for {start_timestamps[i]}-{end_timestamps[i]}\n")
        latest_message = f"\nSystem: Here's the video for {start_timestamps[i]}-{end_timestamps[i]}\n"
        file = genai.get_file(name=video_file.name)
        print(f"Retrieved file '{file.display_name}'")
        text_response = text_model.generate_content(
            [initial_prompt + last_message + latest_message, file]
        )
        for candidate in text_response.candidates:
            for part in candidate.content.parts:
                if 'text' in part:
                    latest_message += f"Assistant: {part.text}\n"
        convo_history += latest_message
        last_message = latest_message
        print(latest_message)
        tool_response = tool_model.generate_content(
            [initial_prompt+latest_message+'\nSystem: Please submit ALL the timestamps for Type: NEW SLIDE. If you have decided not to submit any timestamps, please just submit an empty list [].'],
            tool_config={'function_calling_config': 'ANY'}
        )
        tool_message = ""
        for candidate in tool_response.candidates:
            for part in candidate.content.parts:
                if 'function_call' in part:
                    fc = part.function_call
                    assert fc.name == 'submit_slide_transitions'
                    predicted_timestamps = fc.args['timestamps']
                    concluding_slide = False
                    if 'concluding_slide' in fc.args:
                        concluding_slide = fc.args['concluding_slide']
                    if concluding_slide:
                        tool_message += f"Internal Tool Received: {fc.name}({predicted_timestamps}, concluding_slide=True)\n"
                    else:
                        tool_message += f"Internal Tool Received: {fc.name}({predicted_timestamps})\n"
                    entire_video_predicted_timestamps.extend(predicted_timestamps)
        print(tool_message)
        convo_history += tool_message
        genai.delete_file(file.name)
        if concluding_slide or "END OF LECTURE" in latest_message:
            print("Concluding slide detected. Ending the process.")
            break
        time.sleep(60)
    return {
        'video_id': youtube_id,
        'timestamps': entire_video_predicted_timestamps
    }, {
        'video_id': youtube_id,
        'timestamp_agent_trace': convo_history
    }

