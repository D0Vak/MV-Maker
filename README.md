# MV-Maker

This project generates a simple music video from an audio file. It analyzes the audio (RMS energy and beats) and creates a dynamic visualizer synchronized with the music.

## Requirements

The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Sample Audio (Optional)
If you don't have an audio file, you can generate a sample kick and hi-hat beat:
```bash
python generate_audio.py
```
This will create `data/sample.wav`.

### 2. Create Music Video
Run the main script:
```bash
python main.py
```
A file dialog will appear. Select an audio file (e.g., `data/sample.wav`, or your own `.mp3` / `.wav` / `.m4a` file).
The script will process the audio and generate an MP4 video file in the `output/` directory.

## Directory Structure

- `data/` : Place input audio files here.
- `output/` : Generated video files will be saved here.
- `generate_audio.py` : Script to create a sample beat audio.
- `main.py` : Main script to generate the music video.
- `requirements.txt` : Python dependencies.