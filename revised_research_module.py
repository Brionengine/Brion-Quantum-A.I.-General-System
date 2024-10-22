
import os
import requests
import PyPDF2
import io
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
import pyttsx3
import cv2
import subprocess

# PDF Extraction
def extract_pdf_data(pdf_url):
    response = requests.get(pdf_url)
    pdf_data = io.BytesIO(response.content)
    reader = PyPDF2.PdfReader(pdf_data)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Article Scraper
def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    return " ".join([p.text for p in paragraphs])

# OCR on Images (e.g., scanned books)
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Video Transcription (using a simulation here)
def transcribe_video(video_path):
    audio_file = "audio.wav"
    cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_file}"
    subprocess.call(cmd, shell=True)

    # Simulate transcription for now
    return "Hello from Ryan's transcription engine!"

# Text-to-Audio Conversion
def create_audio_file(text, filename):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Main Research Method to Handle All Sources
def research(task):
    if task.startswith("http"):
        if task.endswith(".pdf"):
            print("Extracting data from PDF...")
            return extract_pdf_data(task)
        else:
            print("Scraping article...")
            return scrape_article(task)
    elif task.endswith((".jpg", ".png")):
        print("Extracting text from image...")
        return extract_text_from_image(task)
    elif task.endswith(".mp4"):
        print("Transcribing video...")
        return transcribe_video(task)
    else:
        return f"No suitable research method found for: {task}"

# Example Usage
if __name__ == "__main__":
    # Example calls to test each feature
    print(research("https://en.wikipedia.org/wiki/Artificial_intelligence"))  # Article scraping
    create_audio_file("Hello from Ryan!", "/mnt/data/test_audio.wav")  # Create sample audio
