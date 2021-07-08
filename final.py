import face_recognition
import os
import cv2
import smtplib
from email.message import EmailMessage
import imghdr


EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASS')
msg = EmailMessage()

def mail():
    msg['Subject'] = 'Criminal Identified'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = 'swaroopudgaonkar@gmail.com'
    msg.set_content('New criminal is identified. An image is attached for your reference.')

known_faces_dir = "dataset"
unknown_faces_dir = "test"
Tolerance = 0.6
Frame_thickness = 3
Font_thickness = 2
MODEL = 'cnn'

print("Loading...")
known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):
    for filename in os.listdir(f"{known_faces_dir}/{name}"):
        image = face_recognition.load_image_file(f"{known_faces_dir}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print("processing...")

for filename in os.listdir(unknown_faces_dir):
    print(filename)
    image = face_recognition.load_image_file(f'{unknown_faces_dir}/{filename}')
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, Tolerance)
        match = None
        if True in results:
            mail()
            with open('test/test.jpg', 'rb') as f:
                file_data = f.read()
                file_type = imghdr.what(f.name)
                file_name = f.name
            msg.add_attachment(file_data, maintype='image', subtype=file_type, filename=file_name)

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                smtp.send_message(msg)
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            #print("Previous Crime: Homicide")
            

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 255, 0]

            cv2.rectangle(image, top_left, bottom_right, color, Frame_thickness)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), Font_thickness)
    cv2.imshow(filename, image)
    cv2.waitKey(10000)
