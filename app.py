import requests
import time
import os
import subprocess
import sys
from flask import Flask, request, render_template_string, redirect, url_for, flash
from openai import OpenAI
from pydub import AudioSegment
import tempfile

# --- Core Transcription Logic ---

ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"
TEMP_DIR = "/tmp"

# --- AssemblyAI Functions ---

def upload_to_assemblyai(api_key, file_path):
    """Uploads a file to AssemblyAI."""
    print(f"--> Uploading file to AssemblyAI: {file_path}")
    headers = {"authorization": api_key}
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{ASSEMBLYAI_BASE_URL}/upload", headers=headers, data=f)
        response.raise_for_status()
        return response.json()["upload_url"]
    except Exception as e:
        print(f"--> ERROR during AssemblyAI upload: {e}", file=sys.stderr)
        return None

def submit_to_assemblyai(api_key, audio_url):
    """Submits the audio for transcription to AssemblyAI."""
    print("--> Submitting to AssemblyAI for transcription...")
    json_data = {"audio_url": audio_url, "speech_model": "universal"}
    headers = {"authorization": api_key, "content-type": "application/json"}
    try:
        response = requests.post(f"{ASSEMBLYAI_BASE_URL}/transcript", json=json_data, headers=headers)
        response.raise_for_status()
        return response.json()["id"]
    except Exception as e:
        print(f"--> ERROR during AssemblyAI submission: {e}", file=sys.stderr)
        return None

def poll_assemblyai(api_key, transcript_id):
    """Polls AssemblyAI for the transcription result."""
    polling_endpoint = f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}"
    headers = {"authorization": api_key}
    print("--> Polling AssemblyAI for results...")
    while True:
        try:
            response = requests.get(polling_endpoint, headers=headers)
            response.raise_for_status()
            result = response.json()
            if result['status'] == 'completed':
                print("--> AssemblyAI transcription complete.")
                return result['text']
            elif result['status'] == 'error':
                print(f"--> ERROR: {result['error']}", file=sys.stderr)
                return None
            else:
                print(f"--> Status: {result['status']}. Waiting...")
                time.sleep(5)
        except Exception as e:
            print(f"--> ERROR while polling AssemblyAI: {e}", file=sys.stderr)
            return None

# --- OpenAI Whisper Function ---

def combine_transcripts_with_overlap(transcripts, overlap_seconds=5):
    """Combines transcripts while handling overlap to avoid duplication."""
    if not transcripts:
        return ""
    
    if len(transcripts) == 1:
        return transcripts[0]
    
    # For now, use simple concatenation with overlap detection
    # In a more sophisticated implementation, you could use fuzzy matching
    # to find overlapping text and remove duplicates
    
    combined = transcripts[0]
    
    for i in range(1, len(transcripts)):
        current_transcript = transcripts[i]
        
        # Simple approach: look for common words at the end of combined
        # and beginning of current transcript
        combined_words = combined.split()
        current_words = current_transcript.split()
        
        # Look for overlap (last 10 words of combined vs first 10 words of current)
        overlap_found = False
        for overlap_size in range(min(10, len(combined_words), len(current_words)), 0, -1):
            combined_end = " ".join(combined_words[-overlap_size:])
            current_start = " ".join(current_words[:overlap_size])
            
            if combined_end.lower() == current_start.lower():
                # Found overlap, remove it from current transcript
                current_transcript = " ".join(current_words[overlap_size:])
                overlap_found = True
                break
        
        # Add current transcript to combined
        if current_transcript.strip():
            combined += " " + current_transcript
    
    return combined

def split_audio_file(file_path, max_size_mb=25, overlap_seconds=5):
    """Splits audio file into chunks smaller than max_size_mb with overlap."""
    try:
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        
        # Calculate chunk duration based on file size
        file_size = os.path.getsize(file_path)
        total_duration = len(audio)
        
        # Calculate chunk duration (in milliseconds)
        chunk_duration = int((max_size_mb * 1024 * 1024 / file_size) * total_duration)
        
        # Ensure minimum chunk duration (30 seconds)
        chunk_duration = max(chunk_duration, 30000)
        
        # Convert overlap to milliseconds
        overlap_ms = overlap_seconds * 1000
        
        # Split audio into chunks with overlap
        chunks = []
        i = 0
        while i < total_duration:
            # Calculate end position for this chunk
            end_pos = min(i + chunk_duration, total_duration)
            
            # Extract chunk
            chunk = audio[i:end_pos]
            if len(chunk) > 0:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            i = end_pos - overlap_ms
            
            # If we're at the end, break
            if i >= total_duration:
                break
        
        return chunks
    except Exception as e:
        print(f"--> ERROR splitting audio file: {e}", file=sys.stderr)
        return None

def transcribe_with_whisper(api_key, file_path):
    """Transcribes audio using OpenAI's Whisper model."""
    print(f"--> Transcribing with Whisper: {file_path}")
    try:
        client = OpenAI(api_key=api_key)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = 25 * 1024 * 1024  # 25MB in bytes
        
        if file_size <= max_size:
            # File is small enough, transcribe directly
            with open(file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            print("--> Whisper transcription complete.")
            return transcript.text
        else:
            # File is too large, split into chunks
            print(f"--> File size ({file_size / 1024 / 1024:.1f}MB) exceeds 25MB limit. Splitting into chunks...")
            chunks = split_audio_file(file_path)
            
            if not chunks:
                return "Error: Failed to split audio file into chunks."
            
            print(f"--> Split into {len(chunks)} chunks for processing...")
            
            # Transcribe each chunk
            all_transcripts = []
            for i, chunk in enumerate(chunks):
                print(f"--> Processing chunk {i+1}/{len(chunks)}...")
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    chunk.export(temp_file.name, format="mp3")
                    temp_path = temp_file.name
                
                try:
                    # Transcribe chunk
                    with open(temp_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    all_transcripts.append(transcript.text)
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Combine all transcripts with overlap handling
            combined_transcript = combine_transcripts_with_overlap(all_transcripts, overlap_seconds=5)
            print("--> Whisper transcription complete (chunked processing).")
            return combined_transcript
            
    except Exception as e:
        print(f"--> ERROR during Whisper transcription: {e}", file=sys.stderr)
        return None

# --- Helper & Main Logic ---

def extract_youtube_title(url):
    """Extracts the title of a YouTube video."""
    try:
        command = ["yt-dlp", "--get-title", "--no-playlist", url]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        title = result.stdout.strip()
        # Clean the title for filename use
        import re
        title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid filename characters
        title = re.sub(r'\s+', '_', title)  # Replace spaces with underscores
        return title[:100]  # Limit length
    except:
        return None

def download_youtube_audio(url, output_path):
    """Downloads audio from a YouTube URL."""
    print(f"--> Downloading audio from: {url}")
    command = ["yt-dlp", "-x", "--audio-format", "mp3", "-o", output_path, url]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("--> Download complete.")
        return output_path
    except FileNotFoundError:
        print("--> ERROR: 'yt-dlp' not found. Please ensure it's installed.", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"--> ERROR: yt-dlp failed:\n{e.stderr}", file=sys.stderr)
        return None

def handle_transcription(api_key, model, file_path=None, url=None):
    """Main logic for both web and CLI."""
    audio_file_path = None
    cleanup_required = False
    original_filename = None

    if url:
        filename = os.path.join(TEMP_DIR, f"youtube_audio_{int(time.time())}.mp3")
        audio_file_path = download_youtube_audio(url, filename)
        if not audio_file_path:
            return "Error: Could not download YouTube audio."
        cleanup_required = True
        # Extract video title from URL or use a default
        original_filename = extract_youtube_title(url) or "youtube_video"
    elif file_path:
        audio_file_path = file_path
        # Get original filename without path and extension
        original_filename = os.path.splitext(os.path.basename(file_path))[0]

    if not audio_file_path or not os.path.exists(audio_file_path):
        return f"Error: File not found at '{audio_file_path}'"

    transcript_text = None
    if model == 'assemblyai':
        upload_url = upload_to_assemblyai(api_key, audio_file_path)
        if upload_url:
            transcript_id = submit_to_assemblyai(api_key, upload_url)
            if transcript_id:
                transcript_text = poll_assemblyai(api_key, transcript_id)
    elif model == 'whisper':
        transcript_text = transcribe_with_whisper(api_key, audio_file_path)
    else:
        return "Error: Invalid model selected."

    if cleanup_required and os.path.exists(audio_file_path):
        print(f"--> Cleaning up temporary file: {audio_file_path}")
        os.remove(audio_file_path)

    # Return both transcript and filename
    if transcript_text and not transcript_text.startswith('Error:'):
        return transcript_text, original_filename
    else:
        return transcript_text or f"Error: Transcription failed with {model}.", None

# --- Flask Web App ---

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()  # Generate random secret key

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <meta http-equiv="X-Content-Type-Options" content="nosniff">
  <meta http-equiv="X-Frame-Options" content="DENY">
  <meta http-equiv="X-XSS-Protection" content="1; mode=block">
  <meta name="referrer" content="strict-origin-when-cross-origin">
  <meta name="theme-color" content="#f5f5f5">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <meta name="msapplication-navbutton-color" content="#f5f5f5">
  <title>Transcriber</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDE0QzEzLjY2IDE0IDE1IDEyLjY2IDE1IDExVjVDMTUgMy4zNCAxMy42NiAyIDEyIDJDMTAuMzQgMiA5IDMuMzQgOSA1VjExQzkgMTIuNjYgMTAuMzQgMTQgMTIgMTRaIiBmaWxsPSIjMDAwIi8+CjxwYXRoIGQ9Ik0xOSAxMVYxMUMxOSAxNC44NyAxNS44NyAxOCAxMiAxOEM4LjEzIDE4IDUgMTQuODcgNSAxMVYxMUg3VjExQzcgMTMuNzYgOS4yNCAxNiAxMiAxNkMxNC43NiAxNiAxNyAxMy43NiAxNyAxMVYxMUgxOVoiIGZpbGw9IiMwMDAiLz4KPHBhdGggZD0iTTEyIDIwQzEyIDIwIDEyIDIyIDEwIDIySDE0QzEyIDIyIDEyIDIwIDEyIDIwWiIgZmlsbD0iIzAwMCIvPgo8L3N2Zz4K">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
      --primary: #000000;
      --primary-hover: #333333;
      --surface: #ffffff;
      --background: #f5f5f5;
      --border: #e0e0e0;
      --text-primary: #000000;
      --text-secondary: #666666;
      --text-light: #999999;
      --danger: #cc0000;
      --warning: #cc6600;
      --success: #006600;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --shadow-sm: none;
      --shadow-md: none;
      --shadow-lg: none;
      --transition: all 0.2s ease;
    }
    
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      background-color: var(--background);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: var(--text-primary);
      min-height: 100vh;
      padding: 20px;
    }
    
    .container {
      max-width: 600px;
      margin: 0 auto;
      padding: 2rem 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .card {
      background: var(--surface);
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      padding: 2rem;
      margin-bottom: 2rem;
      width: 100%;
      transition: var(--transition);
    }
    
    .header {
      text-align: center;
      margin-bottom: 2.8rem;
    }
    
    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 0.6rem;
      letter-spacing: -0.025em;
    }
    
    .header p {
      color: var(--text-secondary);
      font-size: 1.1rem;
      font-weight: 400;
    }
    
    .form-section {
      margin-bottom: 2.2rem;
    }
    
    .form-group {
      margin-bottom: 1.8rem;
    }
    
    .form-label {
      display: block;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 0.6rem;
      font-size: 0.95rem;
    }
    
    .form-input {
      width: 100%;
      padding: 0.9rem 1.1rem;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      font-size: 1rem;
      transition: var(--transition);
      background: var(--surface);
      color: var(--text-primary);
    }
    
    .form-input:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.15);
    }
    
    .form-input::placeholder {
      color: var(--text-light);
    }
    
    .model-selection {
      display: flex;
      gap: 0.8rem;
      margin-bottom: 1.5rem;
    }
    
    .model-option {
      flex: 1;
      position: relative;
    }
    
    .model-option input[type="radio"] {
      display: none;
    }
    
    .model-option label {
      display: block;
      padding: 1rem;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      background: var(--surface);
      cursor: pointer;
      text-align: center;
      font-weight: 500;
      transition: var(--transition);
      color: var(--text-primary);
    }
    
    .model-option input[type="radio"]:checked + label {
      border-color: var(--primary);
      background: rgba(0, 102, 255, 0.05);
      color: var(--primary);
      box-shadow: var(--shadow-sm);
    }
    
    .model-option label:hover {
      border-color: var(--primary);
      transform: translateY(-2px);
    }
    
    .form-select {
      width: 100%;
      padding: 0.9rem 1.1rem;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      font-size: 1rem;
      background: var(--surface);
      cursor: pointer;
      color: var(--text-primary);
      appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%236b7280' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 1rem center;
      background-size: 16px;
      padding-right: 2.5rem;
    }
    
    .form-select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.15);
    }
    
    .file-input-wrapper {
      position: relative;
      display: inline-block;
      width: 100%;
    }
    
    .file-input {
      width: 100%;
      padding: 0.9rem 1.1rem;
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      font-size: 1rem;
      background: var(--surface);
      cursor: pointer;
      color: var(--text-primary);
      transition: var(--transition);
    }
    
    .file-input:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.15);
    }
    
    .divider {
      text-align: center;
      margin: 2.2rem 0;
      position: relative;
      display: flex;
      align-items: center;
    }
    
    .divider::before {
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border);
    }
    
    .divider::after {
      content: '';
      flex: 1;
      height: 1px;
      background: var(--border);
    }
    
    .divider span {
      padding: 0 1.2rem;
      color: var(--text-secondary);
      font-weight: 500;
      font-size: 0.9rem;
      background: var(--surface);
      margin: 0 0.5rem;
    }
    
    .btn {
      display: inline-block;
      padding: 0.9rem 1.5rem;
      border: none;
      border-radius: var(--radius-md);
      font-size: 1rem;
      font-weight: 500;
      cursor: pointer;
      transition: var(--transition);
      text-decoration: none;
      text-align: center;
    }
    
    .btn-primary {
      background: var(--primary);
      color: white;
      width: 100%;
      border: 1px solid var(--primary);
    }
    
    .btn-primary:hover {
      background: var(--primary-hover);
      transform: translateY(-1px);
    }
    
    .btn-primary:active {
      transform: translateY(0);
    }
    

    
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none !important;
    }
    
    .btn-processing {
      position: relative;
      color: transparent !important;
    }
    
    .btn-processing::after {
      content: "Processing...";
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      color: white;
      font-size: 0.9rem;
    }
    
    .processing-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: var(--background);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      display: none;
    }
    
    .processing-content {
      text-align: center;
      max-width: 400px;
      padding: 2rem;
    }
    
    .processing-title {
      font-size: 1.5rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 1rem;
    }
    
    .processing-status {
      color: var(--text-secondary);
      font-size: 1rem;
      margin-bottom: 2rem;
    }
    
    .processing-spinner {
      border: 3px solid rgba(0, 0, 0, 0.1);
      border-top: 3px solid var(--primary);
      border-radius: 50%;
      width: 50px;
      height: 50px;
      animation: spin 1s linear infinite;
      margin: 0 auto 1rem;
    }
    
    .processing-error {
      color: var(--danger);
      font-weight: 600;
    }
    
    .processing-content button {
      margin-top: 1rem;
      padding: 0.8rem 1.5rem;
      border: none;
      border-radius: var(--radius-md);
      background: var(--primary);
      color: white;
      cursor: pointer;
      font-size: 1rem;
      transition: var(--transition);
    }
    
    .processing-content button:hover {
      background: var(--primary-hover);
    }
    
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    .alert {
      padding: 1rem 1.5rem;
      border-radius: var(--radius-md);
      margin-bottom: 1.8rem;
      font-weight: 500;
      display: flex;
      align-items: flex-start;
      border-left: 4px solid transparent;
    }
    
    .alert-danger {
      background: rgba(239, 68, 68, 0.1);
      color: var(--danger);
      border-left-color: var(--danger);
    }
    
    .alert-warning {
      background: rgba(245, 158, 11, 0.1);
      color: var(--warning);
      border-left-color: var(--warning);
    }
    
    .transcript-section {
      margin-top: 2.5rem;
      animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .transcript-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.2rem;
      flex-wrap: wrap;
      gap: 1rem;
    }
    
    .transcript-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-primary);
      margin: 0;
    }
    
    .transcript-actions {
      display: flex;
      gap: 0.6rem;
      flex-wrap: wrap;
    }
    
    .btn-secondary {
      background: rgba(0, 0, 0, 0.05);
      color: var(--text-primary);
      border: 1px solid var(--border);
      padding: 0.6rem 1rem;
      font-size: 0.9rem;
      border-radius: var(--radius-md);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    
    .btn-secondary:hover {
      background: rgba(0, 0, 0, 0.08);
      transform: translateY(-1px);
    }
    
    .btn-secondary:active {
      transform: translateY(0);
    }
    
    .btn-icon {
      margin-right: 0.25rem;
      display: inline-block;
      width: 16px;
      height: 16px;
      vertical-align: text-bottom;
      fill: currentColor;
    }
    
    .transcript-content {
      background: rgba(0, 0, 0, 0.02);
      border: 1px solid var(--border);
      border-radius: var(--radius-md);
      padding: 1.8rem;
      white-space: pre-wrap;
      font-family: 'SF Mono', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 0.95rem;
      line-height: 1.7;
      color: var(--text-primary);
      max-height: 500px;
      overflow-y: auto;
    }
    
    @media (min-width: 768px) {
      .transcript-section {
        max-width: 900px;
        margin: 0 auto;
      }
      
      .transcript-content {
        max-height: 600px;
        font-size: 1rem;
        line-height: 1.8;
      }
    }
    
    @media (min-width: 1024px) {
      .transcript-section {
        max-width: 1000px;
      }
      
      .transcript-content {
        max-height: 700px;
        padding: 2rem;
      }
    }
    
    @media (max-width: 768px) {
      body {
        padding: 10px;
      }
      
      .container {
        padding: 1rem 0.5rem;
        min-height: auto;
      }
      
      .card {
        padding: 1.8rem 1.2rem;
        margin-bottom: 1rem;
      }
      
      .header h1 {
        font-size: 2rem;
      }
      
      .header p {
        font-size: 1rem;
      }
      
      .model-selection {
        flex-direction: column;
        gap: 0.5rem;
      }
      
      .transcript-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.8rem;
      }
      
      .transcript-actions {
        width: 100%;
        justify-content: space-between;
      }
      
      .btn {
        padding: 0.8rem 1.2rem;
      }
      
      .form-input, .form-select, .file-input {
        font-size: 16px; /* Prevents zoom on iOS */
      }
    }
    
    @media (max-width: 480px) {
      body {
        padding: 5px;
      }
      
      .header h1 {
        font-size: 1.8rem;
        margin-bottom: 0.4rem;
      }
      
      .header p {
        font-size: 0.95rem;
      }
      
      .card {
        padding: 1.5rem 1rem;
        border-radius: var(--radius-md);
      }
      
      .btn-secondary {
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        flex: 1;
        justify-content: center;
      }
      
      .transcript-actions {
        gap: 0.4rem;
      }
      
      .transcript-content {
        padding: 1.2rem;
        font-size: 0.9rem;
        max-height: 400px;
      }
    }
    
    @media (max-width: 360px) {
      .header h1 {
        font-size: 1.6rem;
      }
      
      .card {
        padding: 1.2rem 0.8rem;
      }
      
      .btn-secondary {
        padding: 0.4rem 0.6rem;
        font-size: 0.8rem;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <div class="header">
        <h1>Transcriber</h1>
        <p>Transcribe audio files and YouTube videos with AI</p>
      </div>
      
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="alert alert-{{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
      
      {% if show_form %}
      <form method="post" enctype="multipart/form-data" id="transcribe-form">
        <div class="form-section">
          <div class="form-group">
            <label for="api_key" class="form-label">API Key</label>
            <input type="text" class="form-input" id="api_key" name="api_key" placeholder="Enter your API key" required>
          </div>
          
        <div class="form-group">
            <label class="form-label">Transcription Model</label>
            <div class="model-selection">
              <div class="model-option">
                <input type="radio" id="assemblyai" name="model" value="assemblyai" checked>
                <label for="assemblyai">AssemblyAI</label>
              </div>
              <div class="model-option">
                <input type="radio" id="whisper" name="model" value="whisper">
                <label for="whisper">OpenAI Whisper</label>
              </div>
            </div>
          </div>
        </div>
        
        <div class="divider">
          <span>Input Source</span>
        </div>
        
        <div class="form-section">
        <div class="form-group">
            <label for="url" class="form-label">YouTube URL</label>
            <input type="url" class="form-input" id="url" name="url" placeholder="https://www.youtube.com/watch?v=...">
        </div>
          
        <div class="form-group">
            <label for="file" class="form-label">Upload Audio File</label>
            <input type="file" class="file-input" id="file" name="file" accept="audio/*">
          </div>
        </div>
        
        <button type="submit" class="btn btn-primary" id="submit-btn">Transcribe Audio</button>
      </form>
      {% endif %}
      
      <div id="processing-overlay" class="processing-overlay">
        <div class="processing-content">
          <div class="processing-spinner"></div>
          <div class="processing-title">Transcribing Audio</div>
          <div id="processing-status" class="processing-status">Preparing transcription...</div>
        </div>
      </div>
      
      {% if transcript %}
        <div class="transcript-section">
          <div class="transcript-header">
            <h3 class="transcript-title">Transcript</h3>
            <div class="transcript-actions">
              <button type="button" class="btn btn-secondary" onclick="copyToClipboard()">
                <svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
                  <path fill-rule="evenodd" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path>
                  <path fill-rule="evenodd" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path>
                </svg>
                Copy
              </button>
              <button type="button" class="btn btn-secondary" onclick="downloadTranscript()">
                <svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
                  <path fill-rule="evenodd" d="M7.47 10.78a.75.75 0 001.06 0l3.75-3.75a.75.75 0 00-1.06-1.06L8.75 8.44V1.75a.75.75 0 00-1.5 0v6.69L4.78 5.97a.75.75 0 00-1.06 1.06l3.75 3.75zM3.75 13a.75.75 0 000 1.5h8.5a.75.75 0 000-1.5h-8.5z"></path>
                </svg>
                Download
              </button>
              <button type="button" class="btn btn-secondary" onclick="newTranscription()">
                <svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
                  <path fill-rule="evenodd" d="M3.72 3.72a.75.75 0 011.06 0L8 6.94l3.22-3.22a.75.75 0 111.06 1.06L9.06 8l3.22 3.22a.75.75 0 11-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 01-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 010-1.06z"></path>
                </svg>
                New Transcription
              </button>
            </div>
          </div>
          <div class="transcript-content" id="transcript-content">{{ transcript }}</div>
          <script>
            window.transcriptFilename = "{{ filename or 'transcript' }}";
          </script>
        </div>
      {% endif %}
    </div>
  </div>
  
  <script>
    document.getElementById('transcribe-form')?.addEventListener('submit', function() {
      const submitBtn = document.getElementById('submit-btn');
      const processingOverlay = document.getElementById('processing-overlay');
      const processingStatus = document.getElementById('processing-status');
      const processingTitle = document.querySelector('.processing-title');
      
      // Disable button
      submitBtn.disabled = true;
      
      // Show processing overlay
      processingOverlay.style.display = 'flex';
      
      // Update status messages at different intervals
      setTimeout(() => {
        processingStatus.textContent = 'Uploading audio file...';
      }, 1000);
      
      setTimeout(() => {
        processingStatus.textContent = 'Processing with AI model...';
      }, 3000);
      
      setTimeout(() => {
        processingStatus.textContent = 'Generating transcript...';
      }, 6000);
      
      setTimeout(() => {
        processingStatus.textContent = 'Finalizing results...';
      }, 9000);
      
      // Check for errors after form submission
      setTimeout(() => {
        // Look for error messages in the page
        const errorAlerts = document.querySelectorAll('.alert-danger');
        if (errorAlerts.length > 0) {
          // Show error in processing overlay
          processingTitle.textContent = 'Transcription Failed';
          processingStatus.textContent = errorAlerts[0].textContent;
          processingStatus.style.color = 'var(--danger)';
          
          // Hide spinner and show error state
          const spinner = document.querySelector('.processing-spinner');
          if (spinner) {
            spinner.style.display = 'none';
          }
          
          // Add a button to go back
          setTimeout(() => {
            const backButton = document.createElement('button');
            backButton.textContent = 'Try Again';
            backButton.className = 'btn btn-primary';
            backButton.style.marginTop = '1rem';
            backButton.onclick = function() {
              window.location.reload();
            };
            processingStatus.parentNode.appendChild(backButton);
          }, 2000);
        }
      }, 10000); // Check after 10 seconds
    });
    
    function copyToClipboard() {
      const transcriptText = document.getElementById('transcript-content').textContent;
      navigator.clipboard.writeText(transcriptText).then(function() {
        // Show success feedback
        const copyBtn = event.target.closest('.btn');
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<svg class="btn-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16"><path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>Copied!';
        copyBtn.style.background = 'rgba(16, 185, 129, 0.1)';
        copyBtn.style.color = '#10b981';
        copyBtn.style.borderColor = '#10b981';
        setTimeout(() => {
          copyBtn.innerHTML = originalHTML;
          copyBtn.style.background = '';
          copyBtn.style.color = '';
          copyBtn.style.borderColor = '';
        }, 2000);
      }).catch(function(err) {
        console.error('Could not copy text: ', err);
      });
    }
    
    function downloadTranscript() {
      const transcriptText = document.getElementById('transcript-content').textContent;
      const blob = new Blob([transcriptText], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = (window.transcriptFilename || 'transcript') + '.txt';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }
    
    function newTranscription() {
      window.location.href = '/';
    }
  </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    transcript = None
    filename = None
    show_form = True
    
    # Always show form on GET requests (page refresh, direct access)
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, transcript=None, filename=None, show_form=True)
    
    if request.method == 'POST':
        api_key = request.form.get('api_key', '').strip()
        model = request.form.get('model')
        url = request.form.get('url', '').strip()
        file = request.files.get('file')

        # Validate API key format
        if not api_key:
            flash('API Key is required.', 'danger')
            return redirect(url_for('index'))
        
        # Basic API key validation
        if len(api_key) < 20:
            flash('API Key appears to be invalid (too short).', 'danger')
            return redirect(url_for('index'))
        
        if not model:
            flash('Please select a transcription model.', 'danger')
            return redirect(url_for('index'))

        if url:
            # Validate YouTube URL
            if not url.startswith(('https://www.youtube.com/', 'https://youtu.be/', 'https://youtube.com/')):
                flash('Please enter a valid YouTube URL.', 'warning')
                return redirect(url_for('index'))
            result = handle_transcription(api_key, model, url=url)
        elif file and file.filename:
            # Validate file type
            allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'}
            file_ext = os.path.splitext(file.filename.lower())[1]
            if file_ext not in allowed_extensions:
                flash(f'Unsupported file type. Please use: {", ".join(allowed_extensions)}', 'warning')
                return redirect(url_for('index'))
            
            # Validate file size (max 100MB)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            if file_size > 100 * 1024 * 1024:  # 100MB
                flash('File too large. Maximum size is 100MB.', 'warning')
                return redirect(url_for('index'))
            
            filepath = os.path.join(TEMP_DIR, file.filename)
            file.save(filepath)
            result = handle_transcription(api_key, model, file_path=filepath)
            if os.path.exists(filepath):
                os.remove(filepath) # Clean up uploaded file
        else:
            flash('Please provide a YouTube URL or upload a file.', 'warning')
            return redirect(url_for('index'))

        # Handle the new return format
        if isinstance(result, tuple):
            transcript, filename = result
        else:
            transcript = result
            filename = None
        
        # Check if there was an error
        if transcript and transcript.startswith('Error:'):
            flash(transcript, 'danger')
            return redirect(url_for('index'))
        elif transcript:
            show_form = False

    return render_template_string(HTML_TEMPLATE, transcript=transcript, filename=filename, show_form=show_form)

# --- Main Execution ---

if __name__ == "__main__":
    print("--> Starting web server at http://0.0.0.0:3000")
    app.run(host='0.0.0.0', port=3000, debug=True)
