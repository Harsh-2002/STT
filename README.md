# Universal Transcriber Web App

A web application for transcribing audio files and YouTube videos using AssemblyAI or OpenAI Whisper.

## Features

- **Multiple Transcription Models**: Support for AssemblyAI and OpenAI Whisper
- **YouTube Integration**: Direct transcription from YouTube URLs
- **File Upload**: Upload local audio files (MP3, WAV, M4A, OGG)
- **Web Interface**: Clean, responsive web interface

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install yt-dlp** (for YouTube support):
   ```bash
   pip install yt-dlp
   ```

3. **Get API Keys**:
   - **AssemblyAI**: Sign up at [assemblyai.com](https://www.assemblyai.com/)
   - **OpenAI**: Sign up at [openai.com](https://openai.com/)

## Usage

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to:
   ```
   http://localhost:3000
   ```

3. **Use the web interface**:
   - Enter your API key
   - Select transcription model (AssemblyAI or Whisper)
   - Either paste a YouTube URL or upload an audio file
   - Click "Transcribe" and wait for results

## API Keys

- **AssemblyAI**: Use your AssemblyAI API key for the AssemblyAI model
- **OpenAI**: Use your OpenAI API key for the Whisper model

## Supported Audio Formats

- MP3, WAV, M4A, OGG
- YouTube videos (audio will be extracted automatically)

## Notes

- Temporary files are automatically cleaned up after processing
- The app runs on port 3000 by default
- Debug mode is enabled for development 