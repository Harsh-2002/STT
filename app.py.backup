import requests
import time
import os
import subprocess
import sys
from flask import Flask, request, render_template_string, redirect, url_for, flash
from openai import OpenAI

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

def transcribe_with_whisper(api_key, file_path):
    """Transcribes audio using OpenAI's Whisper model."""
    print(f"--> Transcribing with Whisper: {file_path}")
    try:
        client = OpenAI(api_key=api_key)
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print("--> Whisper transcription complete.")
        return transcript.text
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
  <title>Transcriber</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDE0QzEzLjY2IDE0IDE1IDEyLjY2IDE1IDExVjVDMTUgMy4zNCAxMy42NiAyIDEyIDJDMTAuMzQgMiA5IDMuMzQgOSA1VjExQzkgMTIuNjYgMTAuMzQgMTQgMTIgMTRaIiBmaWxsPSIjMDAwIi8+CjxwYXRoIGQ9Ik0xOSAxMVYxMUMxOSAxNC44NyAxNS44NyAxOCAxMiAxOEM4LjEzIDE4IDUgMTQuODcgNSAxMVYxMUg3VjExQzcgMTMuNzYgOS4yNCAxNiAxMiAxNkMxNC43NiAxNiAxNyAxMy43NiAxNyAxMVYxMUgxOVoiIGZpbGw9IiMwMDAiLz4KPHBhdGggZD0iTTEyIDIwQzEyIDIwIDEyIDIyIDEwIDIySDE0QzEyIDIyIDEyIDIwIDEyIDIwWiIgZmlsbD0iIzAwMCIvPgo8L3N2Zz4K">
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    
    body {
      background-color: #f5f5f5;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
    }
    
    .container {
      max-width: 600px;
      margin: 0 auto;
      padding: 2rem 1rem;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    
    .card {
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      padding: 2rem;
      margin-bottom: 2rem;
      width: 100%;
    }
    
    .header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    
    .header h1 {
      font-size: 2.5rem;
      font-weight: 700;
      color: #000;
      margin-bottom: 0.5rem;
    }
    
    .header p {
      color: #666;
      font-size: 1.1rem;
    }
    
    .form-section {
      margin-bottom: 2rem;
    }
    
    .form-group {
      margin-bottom: 1.5rem;
    }
    
    .form-label {
      display: block;
      font-weight: 600;
      color: #000;
      margin-bottom: 0.5rem;
      font-size: 0.95rem;
    }
    
    .form-input {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      font-size: 1rem;
      transition: border-color 0.2s ease;
      background: white;
    }
    
    .form-input:focus {
      outline: none;
      border-color: #000;
    }
    
    .model-selection {
      display: flex;
      gap: 0.5rem;
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
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      background: white;
      cursor: pointer;
      text-align: center;
      font-weight: 600;
      transition: all 0.2s ease;
    }
    
    .model-option input[type="radio"]:checked + label {
      border-color: #000;
      background: #f8f9fa;
    }
    
    .model-option label:hover {
      border-color: #000;
      transform: translateY(-1px);
    }
    
    .form-select {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      font-size: 1rem;
      background: white;
      cursor: pointer;
    }
    
    .form-select:focus {
      outline: none;
      border-color: #000;
    }
    
    .file-input-wrapper {
      position: relative;
      display: inline-block;
      width: 100%;
    }
    
    .file-input {
      width: 100%;
      padding: 0.75rem 1rem;
      border: 2px solid #e0e0e0;
      border-radius: 8px;
      font-size: 1rem;
      background: white;
      cursor: pointer;
    }
    
    .file-input:focus {
      outline: none;
      border-color: #000;
    }
    
    .divider {
      text-align: center;
      margin: 2rem 0;
      position: relative;
      display: flex;
      align-items: center;
    }
    
    .divider::before {
      content: '';
      flex: 1;
      height: 1px;
      background: #e0e0e0;
    }
    
    .divider::after {
      content: '';
      flex: 1;
      height: 1px;
      background: #e0e0e0;
    }
    
    .divider span {
      padding: 0 1rem;
      color: #666;
      font-weight: 600;
      font-size: 0.9rem;
      background: white;
      margin: 0 0.5rem;
    }
    
    .btn {
      display: inline-block;
      padding: 1rem 2rem;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      text-decoration: none;
      text-align: center;
    }
    
    .btn-primary {
      background: #000;
      color: white;
      width: 100%;
    }
    
    .btn-primary:hover {
      background: #333;
      transform: translateY(-1px);
    }
    
    .btn-primary:active {
      transform: translateY(0);
    }
    
    .loader {
      border: 3px solid #f3f3f3;
      border-top: 3px solid #000;
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin: 2rem auto;
      display: none;
    }
    
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    .alert {
      padding: 1rem 1.5rem;
      border-radius: 8px;
      margin-bottom: 1.5rem;
      font-weight: 500;
    }
    
    .alert-danger {
      background: #fee;
      color: #c53030;
      border: 1px solid #fed7d7;
    }
    
    .alert-warning {
      background: #fffbeb;
      color: #c05621;
      border: 1px solid #fef5e7;
    }
    
    .transcript-section {
      margin-top: 2rem;
    }
    
    .transcript-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    
    .transcript-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #000;
      margin: 0;
    }
    
    .transcript-actions {
      display: flex;
      gap: 0.5rem;
    }
    
    .btn-secondary {
      background: #f8f9fa;
      color: #333;
      border: 1px solid #e0e0e0;
      padding: 0.5rem 1rem;
      font-size: 0.9rem;
    }
    
    .btn-secondary:hover {
      background: #e9ecef;
      transform: translateY(-1px);
    }
    
    .transcript-content {
      background: #f8f9fa;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      padding: 1.5rem;
      white-space: pre-wrap;
      font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
      font-size: 0.9rem;
      line-height: 1.6;
      color: #333;
      max-height: 400px;
      overflow-y: auto;
    }
    
    @media (max-width: 768px) {
      .container {
        padding: 1rem 0.5rem;
      }
      
      .card {
        padding: 1.5rem;
      }
      
      .header h1 {
        font-size: 2rem;
      }
      
      .btn {
        padding: 0.875rem 1.5rem;
      }
    }
    
    @media (max-width: 480px) {
      .header h1 {
        font-size: 1.75rem;
      }
      
      .card {
        padding: 1rem;
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
          <span>OR</span>
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
        
        <button type="submit" class="btn btn-primary">Transcribe Audio</button>
      </form>
      {% endif %}
      
      <div id="loader" class="loader"></div>
      
      {% if transcript %}
        <div class="transcript-section">
          <div class="transcript-header">
            <h3 class="transcript-title">Transcript</h3>
            <div class="transcript-actions">
              <button type="button" class="btn btn-secondary" onclick="copyToClipboard()">Copy</button>
              <button type="button" class="btn btn-secondary" onclick="downloadTranscript()">Download</button>
              <button type="button" class="btn btn-secondary" onclick="newTranscription()">New Transcription</button>
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
    document.getElementById('transcribe-form').addEventListener('submit', function() {
      document.getElementById('loader').style.display = 'block';
    });
    
    function copyToClipboard() {
      const transcriptText = document.getElementById('transcript-content').textContent;
      navigator.clipboard.writeText(transcriptText).then(function() {
        // Show success feedback
        const copyBtn = event.target;
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.style.background = '#28a745';
        copyBtn.style.color = 'white';
        setTimeout(() => {
          copyBtn.textContent = originalText;
          copyBtn.style.background = '#f8f9fa';
          copyBtn.style.color = '#333';
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
        
        if transcript and not transcript.startswith('Error:'):
            show_form = False

    return render_template_string(HTML_TEMPLATE, transcript=transcript, filename=filename, show_form=show_form)

# --- Main Execution ---

if __name__ == "__main__":
    print("--> Starting web server at http://0.0.0.0:3000")
    app.run(host='0.0.0.0', port=3000, debug=True)
