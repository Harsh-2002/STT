# Transcriber

A modern web application for transcribing audio files and YouTube videos using AssemblyAI or OpenAI Whisper.

## ✨ Features

- **Multiple Transcription Models**: Support for AssemblyAI and OpenAI Whisper
- **YouTube Integration**: Direct transcription from YouTube URLs with automatic title extraction
- **File Upload**: Upload local audio files (MP3, WAV, M4A, OGG, FLAC, AAC)
- **Smart Filenames**: Downloads use original file/video names
- **Modern UI**: Clean, responsive card-based design
- **Copy & Download**: Copy to clipboard or download as TXT files
- **Security**: Input validation, file size limits, and security headers

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Build the Docker image**:
   ```bash
   docker build -t transcriber .
   ```

2. **Run the container**:
   ```bash
   docker run -p 3000:3000 transcriber
   ```

3. **Access the application**:
   ```
   http://localhost:3000
   ```

### Option 2: Local Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install yt-dlp** (for YouTube support):
   ```bash
   pip install yt-dlp
   ```

3. **Start the web server**:
   ```bash
   python3 app.py
   ```

4. **Access the application**:
   ```
   http://localhost:3000
   ```

## 🔑 API Keys

You'll need API keys for the transcription services:

- **AssemblyAI**: Sign up at [assemblyai.com](https://www.assemblyai.com/)
- **OpenAI**: Sign up at [openai.com](https://openai.com/)

## 📁 Supported Formats

### Audio Files
- MP3, WAV, M4A, OGG, FLAC, AAC
- Maximum file size: 100MB

### YouTube Videos
- All YouTube URLs (youtube.com, youtu.be)
- Automatic audio extraction
- Video title used for download filename

## 🎨 User Interface

### Features
- **Minimal Design**: Clean black and white theme
- **Responsive Layout**: Works on desktop and mobile
- **Card-based UI**: Modern card design with proper spacing
- **Model Selection**: Radio button interface for choosing transcription model
- **Smart Filenames**: Downloads use original file/video names

### Transcript View
- **Copy to Clipboard**: One-click copy with visual feedback
- **Download as TXT**: Automatic download with original filename
- **New Transcription**: Start over with fresh transcription
- **Form Auto-hide**: Form disappears when transcript is shown

## 🔒 Security Features

- **Input Validation**: API key format and length validation
- **File Validation**: Type and size restrictions
- **URL Validation**: YouTube URL format checking
- **Security Headers**: XSS protection, clickjacking prevention
- **HTTPS Ready**: Works with HTTPS proxy (as you mentioned)

## 🐳 Docker

### Build
```bash
docker build -t transcriber .
```

### Run
```bash
docker run -p 3000:3000 transcriber
```

### With Volume (for file persistence)
```bash
docker run -p 3000:3000 -v /host/path:/app/uploads transcriber
```

## 📋 Requirements

- Python 3.11+
- yt-dlp (for YouTube support)
- Flask, requests, openai

## 🔧 Configuration

- **Port**: 3000 (configurable in app.py)
- **Host**: 0.0.0.0 (accessible from any IP)
- **Debug Mode**: Enabled for development
- **Temp Directory**: /tmp (for file processing)

## 🚨 Notes

- **Development Server**: Uses Flask development server (not for production)
- **Temporary Files**: Automatically cleaned up after processing
- **API Keys**: Not stored - entered per session for security
- **Rate Limiting**: Removed for personal use

## 🛠️ Troubleshooting

### Common Issues

1. **"yt-dlp not found"**: Install with `pip install yt-dlp`
2. **"API key invalid"**: Ensure key is at least 20 characters
3. **"File too large"**: Maximum file size is 100MB
4. **"Invalid YouTube URL"**: Use standard YouTube URLs

### Docker Issues

1. **Port already in use**: Change port with `-p 3001:3000`
2. **Permission denied**: Run with `sudo` or check file permissions
3. **Build fails**: Ensure Dockerfile and requirements.txt are in the same directory 