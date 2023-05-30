import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog
import time
import ctypes

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    face_images = []
    unknown_face_names = []

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            # Extract the name from the filename
            name = os.path.splitext(image)[0]
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encodings = face_recognition.face_encodings(face_image)

            if len(face_encodings) > 0:
                # Use only the first face encoding if multiple faces are detected
                face_encoding = face_encodings[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)

        print(self.known_face_names)

    def save_unknown_face(self):
        root.destroy()

    def run_recognition(self):
        start_time = time.time()
        # Vide kaynağı adresi
        video_capture = cv2.VideoCapture(1)
        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        root = tk.Tk()
        save_button = tk.Button(
            root, text="Save", command=self.save_unknown_face)
        save_button.pack()
        file_name_entry = tk.Entry(root)
        file_name_entry.pack()

        while True:
            ret, frame = video_capture.read()
            
            if self.process_current_frame:
            
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations)

                self.face_names = []

                if len(self.face_encodings)== 0:
                    print("yüz bulunamadı")
                    if time.time() - start_time >= 5:
                        # Win+L tuş kombinasyonunu simule etmek için ctypes kullanımı
                            ctypes.windll.user32.LockWorkStation()
                            break

                for face_encoding in self.face_encodings:
                    # Bilindik bir eşleşme var ise
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # En kısa yol hesabı
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(
                            face_distances[best_match_index])
                        if name == "ben":
                            start_time = time.time()
                        else:
                            current_time = time.time()
                         # Eğer 5 saniye boyunca tanınmayan kişi tespit edilmezse
                            if current_time - start_time >= 5:
                        # Win+L tuş kombinasyonunu simule etmek için ctypes kullanımı
                                ctypes.windll.user32.LockWorkStation()
                                break
                    else:
                        current_time = time.time()
                         # Eğer 15 saniye boyunca tanınmayan kişi tespit edilmezse
                        if current_time - start_time >= 5:
                        # Win+L tuş kombinasyonunu simule etmek için ctypes kullanımı
                            ctypes.windll.user32.LockWorkStation()
                            break
                    self.face_names.append(f'{name} ({confidence})')
            self.process_current_frame = not self.process_current_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # if not matches[best_match_index]:
                #     
                #     top = top * 4
                #     right = right * 4
                #     bottom = bottom * 4
                #     left = left * 4

                #     # Bilinmeyen yüz görüntüsünü yakalama
                #     unknown_face_image = frame[top:bottom, left:right]
                #     self.face_images.append(unknown_face_image)

                #     # Kullanıcıdan kayıt adı al
                #     name_entry_text = file_name_entry.get()
                #     self.unknown_face_names.append(name_entry_text)
                # Create the frame with the name
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

          
            cv2.imshow('Face Recognition', frame)

            #  Çıkmak için q'ya basınız
            if cv2.waitKey(1) == ord('q'):
                break

        # Kamerayı serbest bırakmak için gereken komutlar
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
