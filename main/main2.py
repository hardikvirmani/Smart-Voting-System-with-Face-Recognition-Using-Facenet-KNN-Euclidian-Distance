import os
from flask import Flask, render_template, url_for, request, session, flash, redirect
import smtplib
import pymysql
import pandas as pd
import numpy as np
import os
import pickle
import cv2
from PIL import Image
import shutil
import datetime
import time
import requests
import numpy as np
import cv2
import pymysql
from mtcnn import MTCNN
detector = MTCNN()
from keras.models import load_model
from keras import backend as K
from keras.layers import Layer

facenet_model = load_model("facenet_custom_trained.keras")
model = facenet_model
from keras.models import Model
from keras.layers import Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2

def get_facenet_embedding_model():
    base_model = InceptionResNetV2(include_top=False, input_shape=(160, 160, 3), pooling='avg')
    x = Dense(128, activation='linear')(base_model.output)
    return Model(base_model.input, x)





class CustomScaleLayer(Layer):
    def __init__(self, **kwargs):
        self.scale = kwargs.pop('scale', 1.0)
        super(CustomScaleLayer, self).__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, list):
            return [inp * self.scale for inp in inputs]
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CustomScaleLayer, self).get_config()
        config.update({'scale': self.scale})
        return config

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0
    mean, std = img.mean(), img.std()
    img = (img - mean) / std
    return np.expand_dims(img, axis=0)

def preprocess_face_pixels(face_pixels):
    face_pixels = face_pixels.astype('float32') / 255.0
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return np.expand_dims(face_pixels, axis=0)

def get_embedding_from_image(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    sample = np.expand_dims(face_pixels, axis=0) 
    embedding = model.predict(sample)
    return embedding[0]


def get_embeddings(directory, model):
    embeddings = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png')):
                path = os.path.join(root, filename)
                label = os.path.basename(root)
                preprocessed = preprocess_image(path)
                embedding = model.predict(preprocessed)[0]
                embeddings.setdefault(label, []).append(embedding)
    
    for label in embeddings:
        embeddings[label] = np.mean(embeddings[label], axis=0)
    return embeddings

def recognize_face(input_image_path, embeddings, model):
    preprocessed = preprocess_image(input_image_path)
    embedding_input = model.predict(preprocessed)[0].reshape(1, -1)

    similarities = {
        name: cosine_similarity(embedding_input, np.array(db_embedding).reshape(1, -1))[0][0]
        for name, db_embedding in embeddings.items()
    }

    recognized_name = max(similarities, key=similarities.get)
    confidence = similarities[recognized_name]
    return recognized_name, confidence

if __name__ == "__main__":
    model_path = 'facenet_custom_trained.keras'
    images_path = 'embeddings/'


mydb=pymysql.connect(host='localhost', user='root', password='', port=3306, database='smart_voting_system')

# ** Email Sending Configuration **
# sender_address = 'hardikvirmani185@gmail.com' #enter sender's email id
# sender_pass = 'Hardikvirmani@1510' #enter sender's password

app=Flask(__name__)
app.config['SECRET_KEY']='ajsihh98rw3fyes8o3e9ey3w5dc'

@app.before_request
def initialize():
    session['IsAdmin']=False
    session['User']=None

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/admin', methods=['POST','GET'])
def admin():
    if request.method=='POST':
        email = request.form['email']
        password = request.form['password']
        if (email=='admin@voting.com') and (password=='admin'):
            session['IsAdmin']=True
            session['User']='admin'
            flash('Admin login successful','success')
    return render_template('admin.html', admin=session['IsAdmin'])

@app.route('/add_nominee', methods=['GET', 'POST'])
def add_nominee():

    if request.method == 'POST':
        try:
            member = request.form.get('member_name').strip()
            party = request.form.get('party_name').strip()
            logo = request.form.get('test').strip()

            
            if not all([member, party, logo]):
                flash('All fields are required', 'danger')
                return redirect(url_for('add_nominee'))

            cur = mydb.cursor()
            
            
            cur.execute("SELECT member_name FROM nominee WHERE LOWER(member_name) = LOWER(%s)", (member,))
            if cur.fetchone():
                flash(f"Member '{member}' already exists", 'warning')
                return redirect(url_for('add_nominee'))

            
            cur.execute("""
                INSERT INTO nominee (member_name, party_name, symbol_name) 
                VALUES (%s, %s, %s)
            """, (member, party, logo))
            
            mydb.commit()
            flash(f'Successfully added: {member} ({party})', 'success')
            return redirect(url_for('add_nominee'))

        except Exception as e:
            mydb.rollback()
            flash(f'Database error: {str(e)}', 'danger')
            return redirect(url_for('add_nominee'))

        finally:
            cur.close()

    return render_template('nominee.html', admin=True)

@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        state = request.form['state']
        d_name = request.form['d_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        pno = request.form['pno']
        age = int(request.form['age'])
        email = request.form['email']

        
        voters = pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids = voters.aadhar_id.values
        all_voter_ids = voters.voter_id.values

        if age >= 18:
            if (aadhar_id in all_aadhar_ids) or (voter_id in all_voter_ids):
                flash(r'Already Registered as a Voter')
            else:
                
                sql = '''
                INSERT INTO voters 
                (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, verified) 
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                '''
                cur = mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, 'no'))
                mydb.commit()
                cur.close()

                
                session['aadhar'] = aadhar_id
                session['status'] = 'no'
                session['email'] = email

                
                embeddings_path = os.path.join('embeddings', aadhar_id)
                os.makedirs(embeddings_path, exist_ok=True)

                
                return redirect(url_for('capture_images', aadhar_id=aadhar_id))
        else:
            flash("If age is less than 18, not eligible for voting", "info")

    return render_template('voter_reg.html')



@app.route('/verify', methods=['POST', 'GET'])
def verify():
    if 'aadhar' not in session or 'email' not in session:
        flash("Error: No user session found. Register first!", "warning")
        return redirect(url_for('registration'))

    if request.method == 'POST':
        otp_check = request.form['otp_check']
        aadhar_id = session['aadhar']

        
        cur = mydb.cursor()
        cur.execute("SELECT otp FROM voters WHERE aadhar_id = %s", (aadhar_id,))
        stored_otp = cur.fetchone()
        cur.close()

        if stored_otp and otp_check == stored_otp[0]:  
           
            cur = mydb.cursor()
            cur.execute("UPDATE voters SET verified = 'yes', otp = NULL WHERE aadhar_id = %s", (aadhar_id,))
            mydb.commit()
            cur.close()

            flash("Email verified successfully!", "success")
            return redirect(url_for('capture_images'))
        else:
            flash("Incorrect OTP. Please try again.", "danger")
            return redirect(url_for('verify'))

    else:
        
        otp = str(np.random.randint(100000, 999999))
        session['otp'] = otp
        aadhar_id = session['aadhar']

        
        cur = mydb.cursor()
        cur.execute("UPDATE voters SET otp = %s WHERE aadhar_id = %s", (otp, aadhar_id))
        mydb.commit()
        cur.close()

        
        receiver_address = session['email']
        send_email(receiver_address, otp)  

        flash(f"OTP sent to {receiver_address}. Please check your email.", "info")

    return render_template('verify.html')


@app.route('/capture_images', methods=['POST', 'GET'])
def capture_images():
    if 'aadhar' not in session or session['aadhar'] is None:
        flash("Error: No Aadhar ID found. Register first!", "warning")
        return redirect(url_for('registration'))

    aadhar_id = session['aadhar']
    capture_path = os.path.join('embeddings', aadhar_id, 'capture')
    os.makedirs(capture_path, exist_ok=True)

    detector = MTCNN()
    cap = cv2.VideoCapture(0)
    saved = 0
    total_required = 30

    facenet_model = get_facenet_embedding_model()

    while saved < total_required:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector.detect_faces(frame)
        if faces:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            x2, y2 = x + w, y + h
            x2 = min(x2, frame.shape[1])
            y2 = min(y2, frame.shape[0])

            face = frame[y:y2, x:x2]

            # Validate cropped face
            if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                continue

            face_resized = cv2.resize(face, (160, 160))
            face_norm = face_resized.astype('float32') / 255.0
            face_input = np.expand_dims(face_norm, axis=0)

            embedding = facenet_model.predict(face_input, verbose=0)[0]

            np.save(os.path.join(capture_path, f"capture_{saved}.npy"), embedding)
            cv2.imwrite(os.path.join(capture_path, f"capture_{saved}.jpg"), face_resized)

            saved += 1
            print(f"[{saved}/{total_required}] Saved")

            
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

       
        cv2.putText(frame, f"Captured: {saved}/{total_required}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capturing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved >= total_required:
        flash("✅ KNN model trained successfully!", "success")
        return redirect(url_for('train'))  # Or use 'voting' to go straight to face verification

    else:
        flash("⚠️ Not enough faces captured. Try again.", "warning")
        return redirect(url_for('capture_images'))



@app.route('/train')
def train():
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np
    import pickle
    import os
    from flask import flash, redirect, url_for

    X, y = [], []
    dataset_path = "embeddings"

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                try:
                    embedding = np.load(file_path)
                    if embedding.shape == (128,):
                        X.append(embedding)
                        label = file_path.split(os.sep)[1]
                        y.append(label)
                    else:
                        print(f"Skipping {file_path}, invalid shape {embedding.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if len(X) == 0:
        flash("No valid embeddings found to train.", "danger")
        return redirect(url_for('home'))

    X = np.array(X)
    y = np.array(y)

    knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn_model.fit(X, y)

    os.makedirs("train", exist_ok=True)

    with open("train/knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)

    # Optional: save training data
    with open("train/train.pkl", "wb") as f:
        pickle.dump((X, y), f)

    flash("✅Model trained successfully!", "success")
    return redirect(url_for('home')) 



@app.route('/update')
def update():
    return render_template('update.html')
@app.route('/updateback', methods=['POST','GET'])
def updateback():
    if request.method=='POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        middle_name = request.form['middle_name']
        aadhar_id = request.form['aadhar_id']
        voter_id = request.form['voter_id']
        email = request.form['email']
        pno = request.form['pno']
        age = int(request.form['age'])
        voters=pd.read_sql_query('SELECT * FROM voters', mydb)
        all_aadhar_ids=voters.aadhar_id.values
        if age >= 18:
            if (aadhar_id in all_aadhar_ids):
                sql="UPDATE VOTERS SET first_name=%s, middle_name=%s, last_name=%s, voter_id=%s, email=%s,pno=%s, verified=%s where aadhar_id=%s"
                cur=mydb.cursor()
                cur.execute(sql, (first_name, middle_name, last_name, voter_id, email,pno, 'no', aadhar_id))
                mydb.commit()
                cur.close()
                session['aadhar']=aadhar_id
                session['status']='no'
                session['email']=email
                flash(r'Database Updated Successfully','Primary')
                return redirect(url_for('verify'))
            else:
                flash(f"Aadhar: {aadhar_id} doesn't exists in the database for updation", 'warning')
        else:
            flash("age should be 18 or greater than 18 is eligible", "info")
    return render_template('update.html')

@app.route('/voting', methods=['POST', 'GET'])
def voting():
    if request.method == 'POST':
        knn_path = "train/knn_model.pkl"
        facenet_path = "facenet_custom_trained.keras"

        if not os.path.exists(knn_path):
            flash("KNN model not found. Train the model first.", "danger")
            return redirect(url_for('train'))

        if not os.path.exists(facenet_path):
            flash("FaceNet model not found.", "danger")
            return redirect(url_for('home'))

        
        with open(knn_path, "rb") as f:
            knn_model = pickle.load(f)

        facenet_model = load_model(facenet_path)
        detector = MTCNN()

        
        for index in [0, 1, 2]:
            cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cam.isOpened():
                print(f"[INFO] Camera opened at index {index}")
                break
        else:
            flash("Camera access failed. Ensure it's connected and not used by another application.", "danger")
            return redirect(url_for('home'))

        while True:
            ret, frame = cam.read()
            if not ret:
                flash("Camera frame read failed.", "danger")
                break

            frame = cv2.flip(frame, 1)

            
            frame_height, frame_width = frame.shape[:2]
            box_size = 250
            box_x1 = (frame_width - box_size) // 2
            box_y1 = (frame_height - box_size) // 2
            box_x2 = box_x1 + box_size
            box_y2 = box_y1 + box_size

            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
            cv2.putText(frame, "Align face inside the box", (box_x1, box_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            faces = detector.detect_faces(frame)

            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)

                if x < box_x1 or x + w > box_x2 or y < box_y1 or y + h > box_y2:
                    continue

                
                if y + h > frame.shape[0] or x + w > frame.shape[1]:
                    continue

                face_pixels = frame[y:y+h, x:x+w]
                if face_pixels.size == 0:
                    continue

                face_pixels = cv2.resize(face_pixels, (160, 160))
                embedding = get_embedding_from_image(facenet_model, face_pixels)

                prediction = knn_model.predict([embedding])[0]
                confidence = knn_model.predict_proba([embedding]).max()

                
                cursor = mydb.cursor()
                cursor.execute("SELECT first_name, middle_name, last_name FROM voters WHERE aadhar_id = %s", (prediction,))
                result = cursor.fetchone()
                cursor.close()
                name = " ".join(str(part) for part in result if part) if result else "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                label = f"{name} | {prediction} | {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                print(f"[DEBUG] Predicted: {name}, Aadhaar: {prediction}, Confidence: {confidence:.2f}")

                if confidence >= 0.5:
                    session['select_aadhar'] = prediction
                    flash(f"Face Verified!\nAadhaar: {prediction}, Name: {name}, Confidence: {confidence:.2f}", "success")
                    cam.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('select_candidate'))
                else:
                    cv2.putText(frame, "Face not recognized confidently", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Smart Voting - Face Verification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()
        flash("No face recognized. Try again.", "warning")

    return render_template('voting.html')




@app.route('/select_candidate', methods=['POST', 'GET'])
def select_candidate():
    aadhar = session.get('select_aadhar')
    if not aadhar:
        flash("Session expired or unauthorized access. Please try voting again.", "danger")
        return redirect(url_for('voting'))

    df_nom = pd.read_sql_query('SELECT * FROM nominee', mydb)
    all_nom = df_nom['symbol_name'].values

    if request.method == 'POST':
        selected_vote = request.form.get('test')
        if not selected_vote:
            flash("Please select a candidate before submitting.", "danger")
            return render_template('select_candidate.html', noms=sorted(all_nom))

        session['vote'] = selected_vote

        cur = mydb.cursor()

        insert_sql = "INSERT INTO vote (vote, aadhar) VALUES (%s, %s)"
        cur.execute(insert_sql, (selected_vote, aadhar))
        mydb.commit()
        flash("✅ Voted Successfully!", "success")

        cur.close()

        # Corrected the query with parameterized SQL
        voter_df = pd.read_sql_query("SELECT * FROM voters WHERE aadhar_id = %s", mydb, params=(aadhar,))

        if not voter_df.empty:
            phone_no = str(voter_df.iloc[0].get('phone', '9515851969'))
            voter_name = str(voter_df.iloc[0].get('name', 'Voter'))
        else:
            phone_no = '9515851969'
            voter_name = 'Voter'

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        sms_url = "https://www.fast2sms.com/dev/bulkV2"
        message = f"Hi {voter_name}, you voted at {time_stamp} on {date}. Thank you!"

        sms_data = {
            "route": "q",
            "message": message,
            "language": "english",
            "flash": 0,
            "numbers": phone_no,
        }

        headers = {
            "authorization": "UwmaiQR5OoA6lSTz93nP0tDxsFEhI7VJrfKkvYjbM2C14Wde8g9lvA2Ghq5VNCjrZ4THWkF1KOwp3Bxd",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(sms_url, headers=headers, json=sms_data)
            print("[SMS Response]", response.text)
        except Exception as e:
            print("[SMS Error]", str(e))

        return redirect(url_for('home'))

    return render_template('select_candidate.html', noms=sorted(all_nom))



@app.route('/voting_res')
def voting_res():
    votes = pd.read_sql_query('SELECT * FROM vote', mydb)
    
    
    counts = votes['vote'].value_counts().reset_index()
    counts.columns = ['vote_symbol', 'count']  
    all_imgs=['1.png','2.png','3.jpg','4.png','5.png','6.png']
    all_freqs = [counts[counts['vote_symbol'] == i]['count'].values[0] if i in counts['vote_symbol'].values else 0 for i in all_imgs]

    df_nom=pd.read_sql_query('SELECT * FROM nominee', mydb)
    all_nom=df_nom['symbol_name'].values

    return render_template('voting_res.html', freq=all_freqs, noms=all_nom)


if __name__=='__main__':
    app.run(debug=True)
    

