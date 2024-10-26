from flask import Flask, request, jsonify
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from transformers import pipeline
from pydub import AudioSegment
import os
import tempfile
import PyPDF2
import io
import speech_recognition as sr
from io import BytesIO

app = Flask(__name__)

# Configure the Google Generative AI with your API key
genai.configure(api_key="AIzaSyDQYotglq1ftvrrTsZHZ1ms9WC363nesS0")
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    # Get the question and answer from the request body
    data = request.get_json()
    question = data.get("question")
    answer = data.get("answer")

    if not question or not answer:
        return jsonify({"error": "Both question and answer are required"}), 400

    # Construct the prompt with the provided question and answer
    prompt = f"""
    Please evaluate the following answer for its technical correctness. The answer is in response to a specific question, and you should score it on a scale from 0 to 1 based on accuracy, completeness, and relevance to the question. Provide a brief explanation for your score.

    Question: {question}

    Answer: {answer}

    Instructions:
    If the answer is completely correct and thoroughly addresses the question, give it a score of 1.
    If the answer is mostly correct but has minor inaccuracies or missing details, assign a score between 0.6 and 0.9.
    If the answer contains several inaccuracies or lacks key information, give a score between 0.3 and 0.5.
    If the answer is entirely incorrect or irrelevant, give it a score of 0.
    Please provide:

    Score: [Score out of 1]
    Explanation: [Brief explanation of why you assigned this score]
    """

    # Call the Generative AI model
    response = model.generate_content(prompt)

    # Extract and return the response
    if response:
        return jsonify({"response": response.text})
    else:
        return jsonify({"error": "Failed to generate a response"}), 500
    
@app.route('/evaluate_text_quality', methods=['POST'])
def evaluate_text_quality():
    # Get the text to be evaluated from the request body
    data = request.get_json()
    text = data.get("text")

    if not good_text or not text:
        return jsonify({"error": "Both good_text and text are required"}), 400

    # Construct the prompt with the provided good_text and text
    prompt = f"""
    Please evaluate the following text based on three criteria: grammar, vocabulary sophistication, and content richness. 
    Provide scores from 0 to 1 for each criterion and give a brief explanation for each score. Then, calculate a final score 
    as a weighted average (0.4 for grammar, 0.3 for vocabulary, and 0.3 for content richness).

    Text: {text}

    Respond in this format:
    - Grammar Score: [score] - [explanation]
    - Vocabulary Score: [score] - [explanation]
    - Content Richness Score: [score] - [explanation]
    - Final Score: [final score]
    """

    # Call the Generative AI model
    response = model.generate_content(prompt)

    # Extract and return the response
    if response:
        return jsonify({"response": response.text})
    else:
        return jsonify({"error": "Failed to generate a response"}), 500

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

@app.route('/generate_interview_questions', methods=['POST'])
def generate_interview_questions():
    # Check if a PDF file is part of the request
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400
    
    # Read the PDF file
    pdf_file = request.files['pdf']
    resume_text = extract_text_from_pdf(pdf_file)

    # Generate the prompt with the extracted resume text
    prompt = f"""
    I have provided the resume content of a candidate. Analyze it and prepare 10 questions to test their technical skills 
    for the role of software engineer based on the skills and projects mentioned in the resume. The resume content is:
    {resume_text}
    """

    # Call the Generative AI model
    response = model.generate_content(prompt)

    # Return the response
    if response:
        return jsonify({"questions": response.text})
    else:
        return jsonify({"error": "Failed to generate interview questions"}), 500

classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

# Function to extract audio from video
def extract_audio_from_video(video_file):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        # Load video and extract audio
        video_clip = VideoFileClip(video_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(temp_audio_file.name)
        
        # Close video and audio clips
        audio_clip.close()
        video_clip.close()
    
    return temp_audio_file.name

# Function to analyze audio emotion
def analyze_emotion_from_audio(audio_file_path, segment_duration=30000):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file_path)
    
    # Split the audio into segments of specified duration
    segments = [audio[i:i+segment_duration] for i in range(0, len(audio), segment_duration)]
    
    # Predict emotions for each segment
    all_predictions = []
    for i, segment in enumerate(segments):
        # Save the segment as a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as segment_file:
            segment.export(segment_file.name, format="mp3")
            
            # Perform inference on the segment
            predictions = classifier(segment_file.name, top_k=3)  # Top 3 emotions for each segment
            all_predictions.extend(predictions)
            
            # Remove the temporary segment file
            os.remove(segment_file.name)
    
    # Calculate mean scores for each unique emotion
    sum_scores = {}
    count_scores = {}

    for prediction in all_predictions:
        for emotion in prediction:
            label = emotion['label']
            score = emotion['score']
            if label not in sum_scores:
                sum_scores[label] = 0
                count_scores[label] = 0
            sum_scores[label] += score
            count_scores[label] += 1
    
    mean_scores = {emotion: sum_score / count_scores[emotion] for emotion, sum_score in sum_scores.items()}
    return mean_scores

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    # Check if a video file is provided in the request
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']

    try:
        # Extract audio from the video
        audio_file_path = extract_audio_from_video(video_file)

        # Perform emotion analysis on the extracted audio
        emotion_scores = analyze_emotion_from_audio(audio_file_path)

        # Remove the temporary audio file
        os.remove(audio_file_path)

        # Return the emotion analysis results
        return jsonify({"emotions": emotion_scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def transcribe_audio(audio_data):
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    # Load the audio data and transcribe
    with sr.AudioFile(audio_data) as source:
        audio_content = recognizer.record(source)
        # Perform transcription using Google Web Speech API
        text = recognizer.recognize_google(audio_content)
    
    return text

@app.route('/transcribe_audio', methods=['POST'])
def transcribe_audio_endpoint():
    # Check if the audio file is in the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    try:
        # Load the audio file directly from the request without saving it to disk
        audio_data = BytesIO(audio_file.read())
        
        # Transcribe the audio
        transcription = transcribe_audio(audio_data)

        # Return the transcription result
        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


