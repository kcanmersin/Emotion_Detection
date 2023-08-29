import cv2
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'] 

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Classifier")

        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = tf.keras.models.load_model('model_after_fine_tuning.h5')

        self.frame_label = ttk.Label(root)
        self.frame_label.pack()

        self.update_frame()

    def classify_face(self, frame):
        resized_frame = cv2.resize(frame, (48, 48))
        resized_frame = np.expand_dims(resized_frame, axis=0) / 255.0  
        predictions = self.model.predict(resized_frame)[0]
        top3_indices = np.argsort(predictions)[::-1][:3]
        top3_labels = [class_labels[i] for i in top3_indices]
        top3_probs = [predictions[i] for i in top3_indices]
        return top3_labels, top3_probs
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = frame[y:y+h, x:x+w]
                top3_labels, top3_probs = self.classify_face(face)

                for i, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
                    text = f"{label}: {prob:.2%}"
                    cv2.putText(frame, text, (x, y - 30 - 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=image)
            self.frame_label.config(image=self.photo)
            self.frame_label.image = self.photo
        
        self.root.after(10, self.update_frame)

root = tk.Tk()
app = EmotionDetectionApp(root)
root.mainloop()

app.cap.release()
cv2.destroyAllWindows()
