from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import moviepy.editor as mp
import google.generativeai as genai
from PIL import Image
from urllib.parse import quote, unquote  # Import quote and unquote
import json

app = Flask(__name__)

# Configure Google Generative AI
GOOGLE_API_KEY = ""  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def UnderstandVideoByClips(video_path, output_dir, interval=5, batch_size=10):
    video = mp.VideoFileClip(video_path)
    total_frames = int(video.duration / interval)
    results = []

    prompt = """
            You are an expert traffic analysis agent with domain-specific knowledge in transportation systems. 
            Based on the following sequence of video frames from a roadside camera, provide a concise analysis 
            of the traffic scenario. Include:

            Traffic conditions: Identify any congestion or unusual traffic flow patterns. If congestion is 
                                 detected, assess its severity and likely cause.
            Infrastructure status: Check for any visible issues with road infrastructure, such as damaged signs, 
                                    road markings, potholes, or malfunctioning traffic signals. Specify the location 
                                    and probable cause of the issue if found.
            Driving behaviors: Evaluate the behavior of vehicles in the scene. Highlight any abnormal or risky 
                               driving patterns.
            Potential hazards or accidents: Identify any signs of an incident or potential risk. If an accident has 
                                            occurred, provide an initial assessment of its impact on traffic flow 
                                            and suggest actions to minimize disruption.

            Output your findings in a structured JSON format like below, if you can't do, just make a description in sentence what you see in JSON format:

            {
              "traffic_conditions": {
                "congestion": "present or absent",
                "severity": "low or medium or high",
                "cause": "reason for congestion" 
              },
              "infrastructure_status": {
                "issues": "present or absent",
                "description": "describe in details",
              },
              "driving_behaviors": {
                "abnormal_behavior": "present/absent",
                "description": "describe in details"
              },
              "potential_hazards": {
                "hazards": "describe in details",
                "description": "details of the hazard" 
              },
              "comments":{}
            } 
        """

    for i in range(0, total_frames, batch_size):
        batch_frames = []
        for j in range(i, min(i + batch_size, total_frames)):
            t = j * interval
            frame = video.get_frame(t)
            image = Image.fromarray(frame)
            batch_frames.append(image)

        response = model.generate_content([prompt, *batch_frames])
        results.append(response.text)

    with open(os.path.join(output_dir, "result.txt"), 'w', newline='') as file:
        file.write(str(results))

    result_str = results[-1]
    result_str = result_str.replace("`json\n", "").replace("`", "")
    result_json = json.loads(result_str)

    formatted_result = []
    for key, value in result_json.items():
        if isinstance(value, dict):  # Check if the value is a dictionary
            formatted_result.append(f"**{key.replace('_', ' ').title()}:**")
            for inner_key, inner_value in value.items():
                formatted_result.append(f"  - {inner_key.replace('_', ' ').title()}: {inner_value}\n\n")
        else:
            formatted_result.append(f"**{key.replace('_', ' ').title()}:** {value}\n\n")
        formatted_result.append("")  # Add an empty line for spacing
    print(formatted_result)
    return "\n".join(formatted_result)  # Join the lines with newline characters



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            output_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(file.filename)[0])
            os.makedirs(output_dir, exist_ok=True)
            try:
                result = UnderstandVideoByClips(filename, output_dir)
                # Quote the result before passing it to the URL
                quoted_result = quote(result)
                return redirect(url_for('result', filename=file.filename, result=quoted_result))
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    return render_template('index.html')

@app.route('/result/<filename>/<result>')
def result(filename, result):
    # Unquote the result to get the original string
    result = unquote(result)
    return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=True)
