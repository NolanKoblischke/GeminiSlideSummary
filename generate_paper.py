import json
import os
import google.generativeai as genai
import cv2
from PIL import Image

def generate_paper(video_id, directory):
    # Configuration for the Gemini API
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-exp-0801")

    # Directory and video ID setup
    slides_dir = os.path.join(directory, 'slides', video_id)

    # Load the JSON files
    timestamps_path = os.path.join(directory, 'predicted_timestamps.json')
    transcripts_path = os.path.join(directory, 'transcripts.json')
    timestamps = json.load(open(timestamps_path))
    transcripts = json.load(open(transcripts_path))

    # Filter for the specific video ID
    timestamps = [x for x in timestamps if x['video_id'] == video_id][0]['timestamps']
    transcript = [x for x in transcripts if x['video_id'] == video_id][0]['transcript']
    # Define the directory containing the images
    # Get all .png files in the directory
    image_files = [f for f in os.listdir(slides_dir) if f.endswith('.png')]
    #sort
    image_files = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    slides = []
    for image_file in image_files:
        # Load the image using cv2
        image_path = os.path.join(slides_dir, image_file)
        image = cv2.imread(image_path)
        
        # Define the font, scale, and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1  # Adjust the scale to increase or decrease the font size
        font_thickness = 2
        text_color = (0, 0, 255)  # Red color in BGR format
        
        # Get the text size
        text_size = cv2.getTextSize(image_file, font, font_scale, font_thickness)[0]
        
        # Calculate position for centered text
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = (image.shape[0] + text_size[1]) // 15
        
        # Add the text to the image
        cv2.putText(image, image_file, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
        
        # Convert the image back to PIL format to display in Jupyter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Display the image in the Jupyter notebook
        slides.append(pil_image)
    
    # Prepare the content to be sent to Gemini
    paper_input = """
    Generate an astrophysics paper formatted in HTML. Each slide should be represented in the following format, using a formal academic writing style:
    Do not reference anything as a "Slide" since it is not slides but a paper. However every slide must be represented in the paper.
    Do not just say "This section is about..." but rather write the content of the slide in a coherent and scientific manner. Make it flow.
    The user is reading a paper, not watching a presentation. The slides are just a reference for the content of the paper.
    Make reference to figures in the slides in your text, but dont reference them by a name or a slide, just say "The left figure shows...".
    Example Slide:
    <div class="slide">
        <h2>The Challenge of Resolution in Cosmological Simulations</h2>
        <p>Cosmological simulations have long been fraught by resolution errors, particularly when modeling the small-scale structure of the universe. These errors often lead to inaccurate representations of dark matter halos and galaxy formation processes. Recent advances in computational power and algorithm efficiency have allowed for higher resolution simulations, which in turn provide more accurate predictions of the large-scale structure of the cosmos. The figure shows that computational cost increases dramatically for finer simulation resolutions, which demonstrates that challenges remain in balancing resolution with computational feasibility.</p>
        <img src="slide_00_00.png" alt="Image 1">
    </div>

    Use this format for all slides, and ensure that the content is synthesized into a coherent and scientifically rigorous narrative.

    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>(INSERT TITLE HERE GEMINI)</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
            }
            .content {
                max-width: 800px;
                margin: auto;
            }
            .slide {
                margin-bottom: 40px;
            }
            .slide img {
                display: block;
                margin: 0 auto 20px auto;
                max-width: 100%;
            }
            .slide p {
                text-align: center;
            }
            h1, h2 {
                color: #2c3e50;
                text-align: left;
            }
            .content p {
                text-align: left;
                margin: 0 20px;
            }
        </style>
    </head>
    <body>
        <div class="content">

        <h1>(INSERT TITLE HERE)</h1>
        <p><strong>(AUTHOR NAMES IF AVAILABLE)</strong></p>
        <p><em>(ANY OTHER INFO e.g. SUBTITLES OR LOCATION)/em></p>
    """

    # Add content for each slide to the input
    for i in range(len(timestamps)):
        slide_number = i + 1
        slide_filename = f"slide_{timestamps[i].replace(':', '_')}.png"
        if slide_number == 1:
            start = '00:00'
            end = timestamps[i]
        elif slide_number == len(timestamps):
            start = timestamps[i - 1]
            end = transcript[-1]['timestamp']
        else:
            start = timestamps[i - 1]
            end = timestamps[i]
        
        # Extract everything said during the slide
        said_during_slide = ' '.join([x['text'] for x in transcript if x['timestamp'] >= start and x['timestamp'] <= end])
        
        # Add the slide content and image reference to the input
        paper_input += f"""
        Slide {slide_number} {start} {slide_filename}
        {said_during_slide}

        Please include {slide_filename} in the flow of the paper, and include it in the HTML output.
        """

    # Close the HTML content (Gemini will fill in the summaries)
    paper_input += """
        </div>
    </body>
    </html>
    """

    # Prompt Gemini to generate the HTML paper with summaries
    response = model.generate_content([paper_input]+slides)

    # Save the generated HTML content to a file
    # output_html = 'astrophysics_paper.html'
    #slides_dir/output_html
    output_html = os.path.join(slides_dir, f'paper_{video_id}.html')
    with open(output_html, 'w') as f:
        f.write(response.candidates[0].content.parts[0].text.strip())

    print(f"Generated paper saved to {output_html}")