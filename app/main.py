import psycopg2
import numpy as np
from typing import List, Dict, Optional
from psycopg2.extras import execute_values
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from .face_processor import FaceProcessor
from .face_vector import FaceEmbeddingDB
import cv2
import tempfile
import os
import face_recognition
import math
from fastapi import HTTPException
import hashlib
import re
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

MAX_DISTANCE = os.getenv('MAX_DISTANCE')
if not MAX_DISTANCE:
    raise ValueError("Missing required environment variable: MAX_DISTANCE")
MAX_DISTANCE = int(MAX_DISTANCE)

ADMIN_PASSCODE = os.getenv('ADMIN_PASSCODE')
if not ADMIN_PASSCODE:
    raise ValueError("Missing required environment variable: ADMIN_PASSCODE")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Validate required environment variables
required_env_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize database handler
db_handler = FaceEmbeddingDB(db_params)

# Initialize face processor
face_processor = FaceProcessor(db_handler)
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

@app.post("/branch/add")
async def add_branch(
    branch_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """
    Add a new branch with its geolocation.

    Args:
        branch_name: Name of the branch
        latitude: Latitude of the branch location
        longitude: Longitude of the branch location
    
    Returns:
        Branch ID and status message
    """
    try:
        branch_id = db_handler.add_branch(branch_name, latitude, longitude)
        if branch_id:
            return {
                "status": "success",
                "message": f"Branch '{branch_name}' added successfully",
                "branch_id": branch_id
            }
        else:
            return {"status": "error", "message": "Failed to add branch"}
    except Exception as e:
        return {"status": "error", "message": f"Error adding branch: {str(e)}"}

@app.get("/branches")
async def get_branches():
    """Get all branches with their locations."""
    try:
        branches = db_handler.get_branches()
        return {"status": "success", "branches": branches}
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving branches: {str(e)}"}

@app.post("/enroll-employee")
async def enroll_employee(
    entity_id: str = Form(...),
    name: str = Form(...),
    branch_id: int = Form(...),
    photo: UploadFile = File(...)
):
    """
    Enroll an employee with photo, entity ID, and branch assignment.

    Args:
        entity_id: Unique identifier for the employee
        name: Name of the employee
        branch_id: Branch ID where employee works
        photo: Employee photo for facial recognition

    Returns:
        Status of enrollment operation
    """
    try:
        # Input validation
        if not entity_id or entity_id.strip() == "":
            return {"status": "error", "message": "Entity ID cannot be empty"}

        if not name or name.strip() == "":
            return {"status": "error", "message": "Name cannot be empty"}

        # Clean inputs
        entity_id = entity_id.strip()
        name = name.strip()

        # Define the directory path
        employee_images_dir = r"employee_images"
        os.makedirs(employee_images_dir, exist_ok=True)
        employee_dir = os.path.join(employee_images_dir, entity_id)
        os.makedirs(employee_dir, exist_ok=True)

        # Define the path for the photo
        photo_path = os.path.join(employee_dir, f"{entity_id}.jpg")

        # Save the uploaded photo to the specified path
        with open(photo_path, "wb") as f:
            content = await photo.read()
            f.write(content)

        # Generate embedding
        image = face_recognition.load_image_file(photo_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            return {"status": "error", "message": "No face detected in the image"}
        
        encoding = face_encodings[0]

        # Check if a similar embedding already exists
        similar_embeddings = db_handler.vector_search(encoding)
        if similar_embeddings:
            return {"status": "error", "message": "Employee with similar face already exists"}

        # Store the embedding in the database with branch assignment
        success = db_handler.store_embedding(entity_id, name, encoding, branch_id)
        if success:
            return {
                "status": "success",
                "message": f"Employee {name} with ID {entity_id} enrolled successfully"
            }
        else:
            return {"status": "error", "message": "Failed to store employee data"}

    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}
    
@app.post("/verify-face")
async def verify_face(
    photo: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """
    Verify a face against stored embeddings and check location against employee's branch.

    Args:
        photo: The photo to verify
        latitude: Current latitude of the employee
        longitude: Current longitude of the employee

    Returns:
        Verification status and employee details if successful
    """
    try:
        # Save the uploaded photo to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_photo:
            content = await photo.read()
            temp_photo.write(content)
            temp_photo_path = temp_photo.name

        # Process the photo to detect faces
        image = cv2.imread(temp_photo_path)
        
        # Clean up the temporary file
        os.unlink(temp_photo_path)
        
        if image is None:
            return {"status": "error", "message": "Invalid image file"}
        
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {"status": "error", "message": "No face detected in the image"}
            
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            return {"status": "error", "message": "Could not generate face encoding"}
        
        # Get the first face encoding
        face_encoding = face_encodings[0]
        
        # Search for similar embeddings
        results = db_handler.vector_search(face_encoding)
        if not results:
            return {"status": "error", "message": "Face not recognized"}
        
        # Get the best match
        best_match = results[0] 
        entity_id = best_match["entity_id"]
        name = best_match["name"]
        
        # Get branch location for this employee
        employee_info = db_handler.get_employee_branch_location(entity_id)
        if not employee_info:
            return {"status": "error", "message": "Employee branch information not found"}
        
        # Check if employee is within the geofence of their branch
        branch_latitude = employee_info["latitude"]
        branch_longitude = employee_info["longitude"]
        
        distance = haversine(latitude, longitude, branch_latitude, branch_longitude)
        if distance > MAX_DISTANCE:
            return {
                "status": "error",
                "message": f"Location verification failed. You are {int(distance)}m away from your branch."
            }
            
        # All checks passed, return success
        return {
            "status": "success",
            "message": "Face and location verified successfully",
            "employee": {
                "entity_id": entity_id,
                "name": name,
                "branch": employee_info["branch_name"],
                "distance": int(distance)
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.post("/process-checkin")
async def process_checkin(
    photo: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """
    Process check-in photo and validate against employee's branch location.
    
    Args:
        photo: The photo to verify
        latitude: Current latitude of the employee
        longitude: Current longitude of the employee
        
    Returns:
        Check-in status and employee details if successful
    """
    try:
        # Save the uploaded photo to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_photo:
            content = await photo.read()
            temp_photo.write(content)
            temp_photo_path = temp_photo.name

        # Process the photo to detect faces
        image = cv2.imread(temp_photo_path)
        
        # Clean up the temporary file
        os.unlink(temp_photo_path)
        
        if image is None:
            return {"status": "error", "message": "Invalid image file"}
        
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {"status": "error", "message": "No face detected in the image"}
            
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            return {"status": "error", "message": "Could not generate face encoding"}
        
        # Get the first face encoding
        face_encoding = face_encodings[0]
        
        # Search for similar embeddings
        results = db_handler.vector_search(face_encoding)
        if not results:
            return {"status": "error", "message": "Face not recognized"}
        
        # Get the best match
        best_match = results[0] 
        entity_id = best_match["entity_id"]
        name = best_match["name"]
        
        # Get branch location for this employee
        employee_info = db_handler.get_employee_branch_location(entity_id)
        if not employee_info:
            return {"status": "error", "message": "Employee branch information not found"}
        
        # Check if employee is within the geofence of their branch
        branch_latitude = employee_info["latitude"]
        branch_longitude = employee_info["longitude"]
        
        distance = haversine(latitude, longitude, branch_latitude, branch_longitude)
        if distance > MAX_DISTANCE:
            return {
                "status": "error",
                "message": f"Location verification failed. You are {int(distance)}m away from your branch."
            }
        
        # Check if the employee has already checked in today
        if db_handler.has_checked_in_today(entity_id):
            return {
                "status": "error", 
                "message": f"{name} has already checked in today"
            }

        # Log the attendance
        success = db_handler.log_attendance(entity_id, 'checkin', latitude, longitude)
        if not success:
            return {"status": "error", "message": "Failed to log check-in"}
            
        # All checks passed, return success
        return {
            "status": "success",
            "message": f"{name} checked in successfully!",
            "employee": {
                "entity_id": entity_id,
                "name": name,
                "branch": employee_info["branch_name"],
                "distance": int(distance)
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error during check-in: {str(e)}"}

@app.post("/process-checkout")
async def process_checkout(
    photo: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """
    Process check-out photo and validate against employee's branch location.
    
    Args:
        photo: The photo to verify
        latitude: Current latitude of the employee
        longitude: Current longitude of the employee
        
    Returns:
        Check-out status and employee details if successful
    """
    try:
        # Save the uploaded photo to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_photo:
            content = await photo.read()
            temp_photo.write(content)
            temp_photo_path = temp_photo.name

        # Process the photo to detect faces
        image = cv2.imread(temp_photo_path)
        
        # Clean up the temporary file
        os.unlink(temp_photo_path)
        
        if image is None:
            return {"status": "error", "message": "Invalid image file"}
        
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return {"status": "error", "message": "No face detected in the image"}
            
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if not face_encodings:
            return {"status": "error", "message": "Could not generate face encoding"}
        
        # Get the first face encoding
        face_encoding = face_encodings[0]
        
        # Search for similar embeddings
        results = db_handler.vector_search(face_encoding)
        if not results:
            return {"status": "error", "message": "Face not recognized"}
        
        # Get the best match
        best_match = results[0] 
        entity_id = best_match["entity_id"]
        name = best_match["name"]
        
        # Get branch location for this employee
        employee_info = db_handler.get_employee_branch_location(entity_id)
        if not employee_info:
            return {"status": "error", "message": "Employee branch information not found"}
        
        # Check if employee is within the geofence of their branch
        branch_latitude = employee_info["latitude"]
        branch_longitude = employee_info["longitude"]
        
        distance = haversine(latitude, longitude, branch_latitude, branch_longitude)
        if distance > MAX_DISTANCE:
            return {
                "status": "error",
                "message": f"Location verification failed. You are {int(distance)}m away from your branch."
            }
        
        # Check if the employee has already checked out today
        if db_handler.has_checked_out_today(entity_id):
            return {
                "status": "error", 
                "message": f"{name} has already checked out today"
            }

        # Log the attendance
        success = db_handler.log_attendance(entity_id, 'checkout', latitude, longitude)
        if not success:
            return {"status": "error", "message": "Failed to log check-out"}
            
        # All checks passed, return success
        return {
            "status": "success",
            "message": f"{name} checked out successfully!",
            "employee": {
                "entity_id": entity_id,
                "name": name,
                "branch": employee_info["branch_name"],
                "distance": int(distance)
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error during check-out: {str(e)}"}
    
@app.get("/employees")
async def get_employees():
    """
    Get a list of all employees with their branch assignments.
    
    Returns:
        List of employees with their details.
    """
    try:
        employees = db_handler.get_all_employees()
        if employees:
            return {
                "status": "success",
                "employees": employees,
                "count": len(employees)
            }
        else:
            return {
                "status": "success",
                "message": "No employees found",
                "employees": [],
                "count": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving employees: {str(e)}"
        }
    
@app.get("/attendance/{entity_id}")
async def get_attendance(entity_id: str):
    """Get attendance records for a specific employee."""
    try:
        # Modification needed in db_handler to retrieve attendance by entity_id
        attendance_records = db_handler.retrieve_attendance(entity_id)
        return {"status": "success", "entity_id": entity_id, "attendance_records": attendance_records}
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving attendance: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.get("/getall")
async def find():
    results = db_handler.retrieve_all_data()
    return results

@app.get("/delete")
async def delete():
    db_handler.delete_tables()
    return {"message": "Tables deleted"}

@app.delete("/delete-user/{name}")
async def delete_user(name: str):
    try:
        success = db_handler.delete_user(name)
        if success:
            return {"status": "success", "message": f"User '{name}' deleted successfully"}
        else:
            return {"status": "error", "message": f"User '{name}' not found"}
    except Exception as e:
        return {"status": "error", "message": f"Error deleting user: {str(e)}"}

@app.get("/attendance/{user_name}")
async def get_attendance(user_name: str):
    """Fetch attendance records for a specific user."""
    attendance_records = db_handler.retrieve_attendance(user_name)
    return attendance_records

@app.get("/user-report/{user_name}")
async def get_user_report(user_name: str):
    """
    Get detailed attendance report for a specific user.

    Args:
        user_name: The name of the user to get the report for

    Returns:
        List of daily attendance records with check-in and check-out times
    """
    try:
        report = db_handler.get_user_attendance_report(user_name)
        if report:
            return {
                "status": "success",
                "user_name": user_name,
                "attendance_records": report
            }
        else:
            return {
                "status": "error",
                "message": f"No attendance records found for user '{user_name}'"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving attendance report: {str(e)}"
        }

@app.get("/notfound")
async def notfound():
    return {"message": "Table not found"}

@app.get("/user-report/{entity_id}")
async def get_user_report(entity_id: str):
    """
    Get detailed attendance report for a specific employee.

    Args:
        entity_id: The entity ID of the employee

    Returns:
        List of daily attendance records with check-in and check-out times
    """
    try:
        report = db_handler.get_user_attendance_report(entity_id)
        if report:
            return {
                "status": "success",
                "entity_id": entity_id,
                "attendance_records": report
            }
        else:
            return {
                "status": "error",
                "message": f"No attendance records found for employee with ID '{entity_id}'"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving attendance report: {str(e)}"
        }
        
@app.post("/create-admin")
async def create_admin(email: str = Form(...), passcode: str = Form(...)):
    """
    Create a new admin by sending a verification email.
    
    Args:
        email: Email address for the new admin
        passcode: Security passcode to authorize admin creation
        
    Returns:
        Status of admin creation request
    """
    try:
        # Verify passcode
        if passcode != ADMIN_PASSCODE:
            raise HTTPException(status_code=401, detail="Invalid passcode")
        
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return {"status": "error", "message": "Invalid email format"}
        
        token = db_handler.create_admin_verification_token(email)
        if not token:
            return {"status": "error", "message": "Failed to create verification token"}
        
        email_sent = face_processor.send_admin_verification_email(email, token)
        if not email_sent:
            return {"status": "error", "message": "Failed to send verification email"}
            
        return {
            "status": "success",
            "message": f"Verification email sent to {email}. Please check your email to complete admin setup."
        }
    except HTTPException as e:
        return {"status": "error", "message": e.detail, "status_code": e.status_code}
    except Exception as e:
        return {"status": "error", "message": f"Error creating admin: {str(e)}"}

@app.post("/verify-admin-token")
async def verify_admin(
    token: str = Form(...),
    username: str = Form(...),
    password: str = Form(...)
):
    """
    Verify admin token and create admin account.
    
    Args:
        token: Verification token from email
        username: Chosen username for admin
        password: Chosen password for admin
        
    Returns:
        Status of admin account creation
    """
    try:
        # Input validation
        if not token or not username or not password:
            return {"status": "error", "message": "All fields are required"}
        
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        success = db_handler.verify_admin_token(token, username, password_hash)
        if success:
            return {
                "status": "success",
                "message": "Admin account created successfully. You can now log in."
            }
        else:
            return {
                "status": "error",
                "message": "Invalid or expired token. Please request a new admin invitation."
            }
    except Exception as e:
        return {"status": "error", "message": f"Error verifying admin token: {str(e)}"}

@app.post("/admin-login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    """
    Verify admin login credentials.
    
    Args:
        username: Admin username
        password: Admin password
        
    Returns:
        Login status
    """
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        is_valid = db_handler.verify_admin_credentials(username, password_hash)
        if is_valid:
            return {
                "status": "success",
                "message": "Login successful"
            }
        else:
            return {
                "status": "error",
                "message": "Invalid username or password"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error during login: {str(e)}"}
    
    
    
    
    
    
    

@app.get("/attendance/today/summary")
async def today_attendance_summary():
    """
    Get summary of today's attendance and weekly average.
    
    Returns:
        Count of employees checked in today and weekly average percentage
    """
    try:
        # Get today's date
        today = datetime.now().date()
        
        # Query for today's attendance
        with db_handler.conn.cursor() as cur:
            # Get total employees count
            cur.execute("SELECT COUNT(*) FROM face_embeddings")
            total_employees = cur.fetchone()[0]
            
            if total_employees == 0:
                return {
                    "status": "success", 
                    "count": 0, 
                    "percentage": 0,
                    "weeklyAverage": 0
                }
            
            # Count unique employees who checked in today
            cur.execute("""
                SELECT COUNT(DISTINCT entity_id) 
                FROM attendance 
                WHERE event_type = 'checkin' AND DATE(event_time) = %s
            """, (today,))
            today_count = cur.fetchone()[0]
            
            # Calculate today's percentage
            today_percentage = (today_count / total_employees) * 100 if total_employees > 0 else 0
            
            # Get weekly average (last 7 days)
            one_week_ago = today - timedelta(days=7)
            cur.execute("""
                WITH daily_counts AS (
                    SELECT 
                        DATE(event_time) as attendance_date,
                        COUNT(DISTINCT entity_id) as daily_count
                    FROM attendance
                    WHERE 
                        event_type = 'checkin' AND 
                        DATE(event_time) BETWEEN %s AND %s
                    GROUP BY DATE(event_time)
                )
                SELECT AVG(daily_count) as weekly_avg
                FROM daily_counts
            """, (one_week_ago, today))
            
            result = cur.fetchone()
            weekly_avg_count = result[0] if result and result[0] is not None else 0
            weekly_avg_percentage = (weekly_avg_count / total_employees) * 100 if total_employees > 0 else 0
            
        return {
            "status": "success",
            "count": today_count,
            "percentage": round(today_percentage, 1),
            "weeklyAverage": round(weekly_avg_percentage, 1)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving attendance summary: {str(e)}"}

@app.get("/attendance/recent-activity")
async def get_recent_activities():
    """
    Get recent attendance and system activities.
    
    Returns:
        List of recent activities with timestamps and details
    """
    try:
        with db_handler.conn.cursor() as cur:
            # Get recent attendance events
            cur.execute("""
                SELECT 
                    a.entity_id,
                    e.name,
                    a.event_type,
                    a.event_time,
                    b.branch_name
                FROM 
                    attendance a
                JOIN 
                    face_embeddings e ON a.entity_id = e.entity_id
                JOIN 
                    branches b ON e.branch_id = b.branch_id
                ORDER BY 
                    a.event_time DESC
                LIMIT 20
            """)
            
            attendance_events = cur.fetchall()
            
            # Format the activities
            activities = []
            for event in attendance_events:
                entity_id, name, event_type, event_time, branch_name = event
                
                if event_type == 'checkin':
                    activity_type = 'check-in'
                    color = 'green'
                    description = f"{name} checked in at {branch_name}"
                else:
                    activity_type = 'check-out'
                    color = 'red'
                    description = f"{name} checked out from {branch_name}"
                
                activities.append({
                    "id": f"{entity_id}-{event_time.isoformat()}",
                    "type": activity_type,
                    "color": color,
                    "description": description,
                    "timestamp": event_time.isoformat(),
                    "timeFormatted": event_time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Also get new employee enrollments (based on creation time in face_embeddings)
            cur.execute("""
                SELECT 
                    entity_id,
                    name,
                    created_at
                FROM 
                    face_embeddings
                ORDER BY 
                    created_at DESC
                LIMIT 5
            """)
            
            enrollments = cur.fetchall()
            
            for enrollment in enrollments:
                entity_id, name, created_at = enrollment
                
                activities.append({
                    "id": f"enroll-{entity_id}",
                    "type": "enrollment",
                    "color": "yellow",
                    "description": f"New employee enrolled: {name}",
                    "timestamp": created_at.isoformat(),
                    "timeFormatted": created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Get new branch additions
            cur.execute("""
                SELECT 
                    branch_id,
                    branch_name,
                    created_at
                FROM 
                    branches
                ORDER BY 
                    created_at DESC
                LIMIT 3
            """)
            
            new_branches = cur.fetchall()
            
            for branch in new_branches:
                branch_id, branch_name, created_at = branch
                
                activities.append({
                    "id": f"branch-{branch_id}",
                    "type": "branch",
                    "color": "blue",
                    "description": f"New branch added: {branch_name}",
                    "timestamp": created_at.isoformat(),
                    "timeFormatted": created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Sort all activities by timestamp
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return {
                "status": "success",
                "activities": activities[:10]  # Return only the 10 most recent
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving recent activities: {str(e)}"}

@app.get("/attendance/report")
async def get_attendance_report(startDate: str = None, endDate: str = None):
    """
    Generate comprehensive attendance report for a date range.
    
    Args:
        startDate: Start date for the report (YYYY-MM-DD)
        endDate: End date for the report (YYYY-MM-DD)
        
    Returns:
        Attendance statistics and daily data for the specified period
    """
    try:
        # Set default dates if not provided
        if not startDate:
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = datetime.strptime(startDate, "%Y-%m-%d")
            
        if not endDate:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(endDate, "%Y-%m-%d")
        
        with db_handler.conn.cursor() as cur:
            # Get total employees count
            cur.execute("SELECT COUNT(*) FROM face_embeddings")
            total_employees = cur.fetchone()[0]
            
            # Get daily attendance counts
            cur.execute("""
                WITH date_range AS (
                    SELECT generate_series(
                        %s::date, 
                        %s::date, 
                        '1 day'::interval
                    )::date AS day
                ),
                daily_counts AS (
                    SELECT 
                        DATE(event_time) as attendance_date,
                        COUNT(DISTINCT entity_id) as check_in_count
                    FROM attendance
                    WHERE 
                        event_type = 'checkin' AND 
                        DATE(event_time) BETWEEN %s AND %s
                    GROUP BY DATE(event_time)
                )
                SELECT 
                    dr.day,
                    COALESCE(dc.check_in_count, 0) as check_in_count
                FROM 
                    date_range dr
                LEFT JOIN 
                    daily_counts dc ON dr.day = dc.attendance_date
                ORDER BY 
                    dr.day
            """, (start_date, end_date, start_date, end_date))
            
            daily_data = cur.fetchall()
            
            # Format daily data
            formatted_daily_data = []
            total_attendance = 0
            
            for day_data in daily_data:
                day, count = day_data
                percentage = (count / total_employees) * 100 if total_employees > 0 else 0
                total_attendance += count
                
                formatted_daily_data.append({
                    "date": day.strftime("%Y-%m-%d"),
                    "count": count,
                    "percentage": round(percentage, 1)
                })
            
            # Calculate summary statistics
            total_days = (end_date - start_date).days + 1
            avg_attendance = total_attendance / total_days if total_days > 0 else 0
            avg_percentage = (avg_attendance / total_employees) * 100 if total_employees > 0 else 0
            
            # Get top branches by attendance
            cur.execute("""
                WITH branch_attendance AS (
                    SELECT 
                        b.branch_id,
                        b.branch_name,
                        COUNT(DISTINCT a.entity_id) as attendance_count
                    FROM 
                        attendance a
                    JOIN 
                        face_embeddings e ON a.entity_id = e.entity_id
                    JOIN 
                        branches b ON e.branch_id = b.branch_id
                    WHERE 
                        a.event_type = 'checkin' AND 
                        DATE(a.event_time) BETWEEN %s AND %s
                    GROUP BY 
                        b.branch_id, b.branch_name
                )
                SELECT 
                    branch_name,
                    attendance_count
                FROM 
                    branch_attendance
                ORDER BY 
                    attendance_count DESC
                LIMIT 5
            """, (start_date, end_date))
            
            top_branches = [{"name": name, "count": count} for name, count in cur.fetchall()]
            
            return {
                "status": "success",
                "summary": {
                    "totalEmployees": total_employees,
                    "avgAttendance": round(avg_attendance, 1),
                    "avgPercentage": round(avg_percentage, 1),
                    "period": {
                        "start": start_date.strftime("%Y-%m-%d"),
                        "end": end_date.strftime("%Y-%m-%d"),
                        "days": total_days
                    },
                    "topBranches": top_branches
                },
                "dailyData": formatted_daily_data
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error generating attendance report: {str(e)}"}