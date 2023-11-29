from flask import Flask, render_template, Response
import cv2
import numpy as np
import base64
import time
import tensorflow as tf
from tensorflow import keras
import platform

app = Flask(__name__)
app.template_folder = 'templates'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = keras.models.load_model('my_model (1).h5')

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio id="audio" autoplay="autoplay" loop="loop">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                var audioElement = document.getElementById('audio');
                audioElement.addEventListener('ended', function() {{
                    this.currentTime = 0;
                    this.play();
                }}, false);
            </script>
            """
        return audio_html

def generate_frames():
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
        
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            centx = w // 2
            centy = h // 2
            eye_1 = roi_color[centy - 40:centy, centx - 70:centx - 10]
            eye_1 = cv2.resize(eye_1, (86, 86))
            eye_2 = roi_color[centy - 40:centy, centx + 10:centx + 70]
            eye_2 = cv2.resize(eye_2, (86, 86))
            
            cv2.rectangle(frame1, (x + centx - 60, y + centy - 40), (x + centx - 10, y + centy), (0, 255, 0), 5)
            cv2.rectangle(frame1, (x + centx + 10, y + centy - 40), (x + centx + 60, y + centy), (0, 255, 0), 5)
            
            preds_eye1 = model.predict(np.expand_dims(eye_1, axis=0))
            preds_eye2 = model.predict(np.expand_dims(eye_2, axis=0))
            e1, e2 = np.argmax(preds_eye1), np.argmax(preds_eye2)
            
            img1 = cv2.imencode('.jpg', frame1)[1].tobytes()
            img2 = cv2.imencode('.jpg', cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))[1].tobytes()
            img3 = cv2.imencode('.jpg', cv2.cvtColor(eye_1, cv2.COLOR_BGR2RGB))[1].tobytes()
            img4 = cv2.imencode('.jpg', cv2.cvtColor(eye_2, cv2.COLOR_BGR2RGB))[1].tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img1 + b'\r\n\r\n'
                   b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img2 + b'\r\n\r\n'
                   b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img3 + b'\r\n\r\n'
                   b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img4 + b'\r\n\r\n')
            
            if e1 != 1 and e2 != 1:
                file_path = "bleep-41488.mp3"
                audio_html = autoplay_audio(file_path)
                yield audio_html.encode()

        except:
            continue
    
    cap.release()

@app.route('/')
def index():
    audio_html = autoplay_audio("bleep-41488.mp3")
    return render_template('index.html', audio_html=audio_html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
