from flask import Flask, render_template, request, jsonify
import cv2
from fer import FER
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/action', methods=['POST'])
def action():
    # Проверяем наличие файла изображения в запросе
    if 'image' not in request.files:
        return jsonify(message="Нет изображения!")

    file = request.files['image']

    if file.filename == '':
        return jsonify(message="Файл не выбран!")

    # Проверяем и создаем папку temp, если она не существует
    if not os.path.exists('temp'):
        os.makedirs('temp')

    # Сохраняем изображение во временный файл
    image_path = os.path.join('temp', file.filename)
    file.save(image_path)

    # Загружаем изображение и анализируем эмоции
    image_one = cv2.imread(image_path)
    emo_detector = FER(mtcnn=True)
    captured_emotions = emo_detector.detect_emotions(image_one)

    if captured_emotions:
        dominant_emotion, emotion_score = emo_detector.top_emotion(image_one)
        os.remove(image_path)  # Удаляем временный файл
        return jsonify(message=f"Доминирующая эмоция: {dominant_emotion}, Оценка: {emotion_score}")
    else:
        os.remove(image_path)  # Удаляем временный файл
        return jsonify(message="Эмоции не обнаружены!")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)




