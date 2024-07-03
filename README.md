# GeminiSlideSummary

## Current Capability

Provide a youtube video of a lecture and it will extract screenshots of unique slides, descriptions of those slides and what the speaker said about them, and a summary of the entire lecture.

## Methodology

1. Download Youtube video
2. Ask Gemini to summarize entire .mp4 video
3. Split video into 10 minute chunks
4. Ask Gemini to extract timestamps of unique slides from each 10 minute .mp4 chunk
    - Skip last 10 minutes since that is usually Q&A
5. Ask Gemini to generate a summary of each timestamp from all of:
   - The 10 minute .mp4 chunk
   - The timestamp
   - A (poor) OCR of a slide screenshot from the timestamp + 5 seconds

## Current Issues

- Timestamp generator still needs somework. Lots of duplicate slides. Maybe duplicates can be identified and deleted after the fact.
- Slide description generator still sometimes mismatching description and slide. Being provided just the audio, full summary, image, and OCR will definitely solve this.
- Sometimes it will fail to extract slide descriptions from Gemini. Need to investigate Gemini's JSON output in these cases. Currently handling this by deleting those slides.
- Sometimes Gemini goes haywire and repeats itself, can add max_tokens or frequency_penalty to prevent this.

## Requirements

- Gemini API Key from [Google AI Studio](aistudio.google.com) set as an environment variable `GEMINI_API_KEY`
  - Gemini 1.5 Flash is free for 1,500 requests per day. A 50 minute video will take 1+5+5=11 requests.
- FFMPEG
- Python modules in requirements.txt (including: moviepy, pytube, easyocr, PIL, google.generativeai)

## Copyright

Please ensure you have the right to use any lecture material summarized by this tool, with permission from the source, respecting copyright and ethical guidelines. The slides copyright belong to the original creator. The developer of this tool is not responsible for any misuse or copyright infringement by users.