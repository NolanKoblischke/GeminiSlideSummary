# GeminiSlideSummary

## Current Capability

Provide a youtube video of a lecture and it will extract timestamps of unique slides and a transcript of the lecture.

## Methodology

1. Download Youtube video
2. Add timer to bottom right corner using cv2 and moviepy
3. Split video into 7-minute segments which Gemini handles better
4. Ask Gemini 1.5 Flash to make a transcript of each segment with timestamps
5. Ask Gemini 1.5 Pro to identify slide changes in each segment and why
6. Ask Gemini 1.5 Flash to read Pro's summary and extract timestamps of new slides only

## Current Issues

- Transcript generation sometimes fails. Could try feeding Gemini only audio instead of video. Could also try supplementing with youtube transcripts that Gemini should improve. Must be done in segments since often transcripts are more than 8192 tokens.

## Requirements

- Gemini API Key from [Google AI Studio](aistudio.google.com) set as an environment variable `GEMINI_API_KEY`
  - Gemini 1.5 Flash is free for 1,500 requests per day.
    - A 50 minute video takes ~14 requests of Flash
  - Gemini 1.5 Pro is free for 50 requests per day.
    - A 50 minute video takes ~7 requests of Pro 
- FFMPEG
- Python modules in requirements.txt (including: moviepy, pytube, easyocr, PIL, google.generativeai)

## Copyright

Please ensure you have the right to use any lecture material summarized by this tool, with permission from the source, respecting copyright and ethical guidelines. The slides copyright belong to the original creator. The developer of this tool is not responsible for any misuse or copyright infringement by users.