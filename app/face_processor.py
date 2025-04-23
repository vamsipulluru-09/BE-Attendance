import face_recognition
import os
import numpy as np
import cv2
from typing import Dict, List, Tuple
from .face_vector import FaceEmbeddingDB
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

APP_URL = os.getenv('APP_URL')
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_PORT = os.getenv('EMAIL_POST')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')


class FaceProcessor:
    def __init__(self,db_handler: FaceEmbeddingDB):
        self.db_handler = db_handler

    def _process_employee_image(self, image_path: str, employee_name: str):
        """Process a single employee image and return its encoding."""
        try:
            # Check if the embedding already exists in the database
            if self.db_handler.embedding_exists(employee_name):
                print(f"Embedding for {employee_name} already exists in the database.")
                return None
            
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                encoding = face_encodings[0]
                self.db_handler.store_embedding(employee_name, encoding)
                return encoding
            return None
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
            
    def process_image(self, image_numpy: np.ndarray) -> List[Dict]:
        """Process a single image and return detected faces with matching logic."""
        # Convert from BGR to RGB if needed
        if len(image_numpy.shape) == 3 and image_numpy.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image_numpy
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        detected_faces = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Get top 5 closest matches from vector search
            results = self.db_handler.vector_search(face_encoding)
            
            name = "Unknown"
            confidence = 0.0
            
            if results:
                try:
                    candidate_encodings = []
                    for result in results:
                        embedding_str = result["embedding"].strip('[]')
                        embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                        candidate_encodings.append(np.array(embedding_values))
                    
                    candidate_names = [result["name"] for result in results]
                    
                    # Compare with candidate faces using face_recognition
                    matches = face_recognition.compare_faces(candidate_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(candidate_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = candidate_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                            # Also consider the vector similarity as a factor
                            vector_confidence = results[best_match_index]["similarity"]
                            # Take the average of both confidence measures
                            confidence = (confidence + vector_confidence) / 2
                except Exception as e:
                    print(f"Error processing embeddings: {e}")
                    continue
            
            # Only include if confidence is high enough
            if name == "Unknown" or confidence > 0.7:
                top, right, bottom, left = face_location
                detected_faces.append({
                    "name": name,
                    "confidence": float(confidence),
                    "location": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    }
                })
        
        return detected_faces
    
    def retrieve_attendance(self, entity_id: str) -> List[Dict[str, any]]:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT a.event_type, a.event_time, a.latitude, a.longitude, e.name
                    FROM attendance a
                    JOIN face_embeddings e ON a.entity_id = e.entity_id
                    WHERE a.entity_id = %s
                    ORDER BY event_time DESC
                """, (entity_id,))
                attendance_records = cur.fetchall()
                return [
                    {
                        "event_type": row[0], 
                        "event_time": row[1],
                        "latitude": row[2],
                        "longitude": row[3],
                        "name": row[4]
                    }
                    for row in attendance_records
                ]
        except Exception as e:
            print(f"Error retrieving attendance: {e}")
            return []

    def get_user_attendance_report(self, entity_id: str) -> List[Dict[str, any]]:
        try:
            with self.conn.cursor() as cur:
                query = """
                    WITH daily_attendance AS (
                        SELECT
                            DATE(event_time) as attendance_date,
                            MAX(CASE WHEN event_type = 'checkin' THEN event_time END) as checkin_time,
                            MAX(CASE WHEN event_type = 'checkout' THEN event_time END) as checkout_time
                        FROM attendance
                        WHERE entity_id = %s
                        GROUP BY DATE(event_time)
                    )
                    SELECT
                        attendance_date,
                        checkin_time,
                        checkout_time
                    FROM daily_attendance
                    ORDER BY attendance_date DESC;
                """
                cur.execute(query, (entity_id,))
                results = cur.fetchall()

                return [
                    {
                        "date": row[0],
                        "checkin_time": row[1],
                        "checkout_time": row[2]
                    }
                    for row in results
                ]
        except Exception as e:
            print(f"Error retrieving user attendance report: {e}")
            return []
        
    def send_admin_verification_email(self, email: str, token: str):
        """Send verification email to new admin."""
        try:
            verification_link = f"{APP_URL}/registration?token={token}"
            
            msg = MIMEMultipart()
            msg['From'] = EMAIL_USERNAME
            msg['To'] = email
            msg['Subject'] = "Admin Account Verification"
            
            body = f"""
            <html>
            <body>
                <h2>Face Recognition Admin Account Setup</h2>
                <p>You have been invited to become an admin for the Face Recognition Attendance System.</p>
                <p>Please click the link below to set up your username and password:</p>
                <p><a href="{verification_link}">Set Up Admin Account</a></p>
                <p>This link will expire in 24 hours.</p>
                <p>If you did not request this invitation, please ignore this email.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Error sending verification email: {e}")
            return False
        