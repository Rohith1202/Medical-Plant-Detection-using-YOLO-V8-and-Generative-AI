#%%writefile CSP_Medplant.py
import os
import streamlit as st
import pandas as pd
import bcrypt
import base64
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import re
import json
import google.generativeai as genai
import cv2
import numpy as np
# Paths for user data
user_data_file = "login_data.csv"
detection_history_file = "Detection History.csv"
feedback_file = "feedback.csv"

# Configure the API key for Gemini
genai.configure(api_key='AIzaSyDGMkXv8Qqh9Bwf2Xs_M6j1UNTSFJC9wBw')  # Replace with your actual API key

# Ensure necessary files exist
def ensure_user_data():
    if not os.path.exists(user_data_file):
        df = pd.DataFrame(columns=['Username', 'Password'])
        df.to_csv(user_data_file, index=False)
def ensure_feedback_file():
    if not os.path.exists(feedback_file):
        pd.DataFrame(columns=["Name", "Age", "Gender", "Rating", "Feedback"]).to_csv(feedback_file, index=False)

ensure_user_data()
ensure_feedback_file()

# Load user data
def load_user_data():
    return pd.read_csv(user_data_file)

# Save new user data
def save_user_data(username, password):
    df = load_user_data()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    new_user = pd.DataFrame([[username, hashed_password]], columns=['Username', 'Password'])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(user_data_file, index=False)

# Check if the username exists
def username_exists(username):
    df = load_user_data()
    return not df[df['Username'] == username].empty

# Validate login
def validate_login(username, password):
    df = load_user_data()
    user_record = df[df['Username'] == username]
    if not user_record.empty:
        stored_hashed_password = user_record['Password'].values[0]
        return bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8'))
    return False

# Change password functionality
def change_password(username, new_password):
    df = load_user_data()
    if username_exists(username):
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        df.loc[df['Username'] == username, 'Password'] = hashed_password
        df.to_csv(user_data_file, index=False)
        return True
    return False

# Load detection history
def load_detection_history():
    if os.path.exists(detection_history_file):
        return pd.read_csv(detection_history_file)
    else:
        return pd.DataFrame(columns=["Name", "Age", "Timestamp", "Purpose", "Detected Plants"])

# Save new detection history entry
def save_detection_history(name, age, purpose, detected_plants):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Timestamp": [timestamp],
        "Purpose": [purpose],
        "Detected Plants": [detected_plants]
    })
    history = load_detection_history()
    history = pd.concat([history, new_entry], ignore_index=True)
    history.to_csv(detection_history_file, index=False)

# Save feedback data to CSV
def save_feedback(name, age, gender, rating, feedback):
    feedback_data = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Rating": [rating],
        "Feedback": [feedback]
    })

    if os.path.exists(feedback_file):
        existing_data = pd.read_csv(feedback_file)
        feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)

    feedback_data.to_csv(feedback_file, index=False)

# Streamlit app title
st.title("Medical Plant Detection Using Deep Learning🪴")

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Display logout message if logged out
if 'logout_message' in st.session_state:
    st.success(st.session_state.logout_message)
    del st.session_state.logout_message  # Clear the message after displaying it

# Display login interface only if not logged in
if not st.session_state.logged_in:
    # Set the background image for the login interface
    def set_login_background(image_file):
        login_bg_img = f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(login_bg_img, unsafe_allow_html=True)

    # Load the background image for the login interface
    with open("medplant loging bg.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background for the login interface
    set_login_background(encoded_image)

    with st.expander("Authentication", expanded=True):
        menu = st.selectbox('Menu', options=['Login', 'Register', 'Forgot Password'])

        if menu == 'Register':
            st.subheader('Register')
            username = st.text_input("Choose a Username", key="register_username")  # Unique key for username
            password = st.text_input("Choose a Password", type="password", key="register_password")  # Unique key for password
            confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")  # Unique key for confirm password

            def is_valid_password(password):
                # Check if password meets the requirements
                if len(password) < 8 or len(password) > 12:
                    st.error("Password must be between 8 to 12 characters length.")
                    return False
                if not any(char.isupper() for char in password):
                    st.error("Password must include at least one uppercase letter (A-Z).")
                    return False
                if not any(char.islower() for char in password):
                    st.error("Password must include at least one lowercase letter (a-z).")
                    return False
                if not any(char.isdigit() for char in password):
                    st.error("Password must include at least one digit (0-9).")
                    return False
                if not re.search(r'[!@#$%^&*]', password):
                    st.error("Password must include at least one special character (!@#$%^&*).")
                    return False
                return True

            if st.button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif username_exists(username):
                    st.error("Username already exists. Please choose a different one.")
                elif not is_valid_password(password):
                    # Password requirements not met, error messages will be displayed in is_valid_password
                    pass
                else:
                    save_user_data(username, password)
                    st.success("Registration successful! You can now log in.")

        elif menu == 'Forgot Password':
            st.subheader('Change Password')
            username = st.text_input("Enter your Username")
            new_password = st.text_input("Enter your New Password", type='password')
            confirm_password = st.text_input("Confirm New Password", type='password')

            if st.button("Change Password"):
                if username_exists(username):
                    if new_password == confirm_password:
                        if change_password(username, new_password):
                            st.success("Your password has been changed successfully.")
                        else:
                            st.error("Failed to change password. Please try again.")
                    else:
                        st.error("Passwords do not match! Please try again.")
                else:
                    st.error("Username not found!")

        elif menu == 'Login':
            st.subheader('Login')
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if validate_login(username, password):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                else:
                    st.error("Invalid username or password")

# Main project interface
if st.session_state.logged_in:
    # Set the background image for the Streamlit interface
    def set_background(image_file):
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_file});
            background-size: cover;
            background-position: center;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

    # Load the background image for the interface
    with open("C:\\Users\\HP\\Downloads\\CSP_Medical-Plant-Detection-Using-Deep-Learning\\medplant bg.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background
    set_background(encoded_image)

    # Main title for the project features
    st.title("Medical Plant Image Detection")

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.logout_message = "You have successfully logged out! Please log in again to continue your exploration of medicinal plant detection."
        st.rerun()  # Refresh the page
    def run_webcam_detection():
        # User inputs for name, age, and purpose of detection
        user_name = st.text_input("Enter your name:")
        user_age = st.text_input("Enter your age:")

        # Dropdown menu for detection purpose
        purpose_options = [
            "Identify Medicinal Properties",
            "Check Plant Authenticity",
            "Identify Edible Plants",
            "Identify Toxic Plants",
            "Determine Plant Dosage",
            "Educational Research",
            "Personal Use",
            "Commercial Use",
            "Others"
        ]
        selected_purpose = st.selectbox("Choose the purpose of detection:", purpose_options)

        

        camera = st.camera_input("Capture an image from your webcam")

        if camera is not None:
            # Ensure upload folder exists
            if not os.path.exists('uploaded_images'):
                os.makedirs('uploaded_images')

            # Save the uploaded image
            filename = Path(camera.name).name
            image_path = os.path.join('uploaded_images', filename)

            with open(image_path, "wb") as f:
                f.write(camera.getbuffer())

            # Display the uploaded image
            #st.image(image_path, caption='Captureded Image', use_column_width=True)

            # Perform detection
            st.write("Processing the image...")
            model_path = "C:\\Users\\HP\\Downloads\\CSP_Medical-Plant-Detection-Using-Deep-Learning\\best.pt"  # Path to your YOLO model
            model = YOLO(model_path)
            results = model.predict(image_path, save=True, save_txt=True)

            # Locate the processed image saved by YOLO
            runs_dir = Path("runs/detect")
            latest_run = max(runs_dir.iterdir(), key=os.path.getmtime)
            processed_image = latest_run / filename

            detected_plant_names = []

            if results:
                for result in results:
                    for box in result.boxes:
                        plant_name = result.names[int(box.cls)]
                        if plant_name not in detected_plant_names:  # Check if the plant name is already in the list
                            detected_plant_names.append(plant_name)
                            break

            if processed_image.exists():
                st.image(str(processed_image), caption='Processed Image', use_column_width=True)

                # Display detected plant names
                if detected_plant_names:
                    st.write("Detected Plants:")
                    for plant in detected_plant_names:
                        st.write(plant)

                    # Save detection history
                    detected_plants = ', '.join(detected_plant_names)
                    save_detection_history(user_name, user_age, selected_purpose, detected_plants)

                    # Load plant details from JSON
                    with open("C:\\Users\\HP\\Downloads\\CSP_Medical-Plant-Detection-Using-Deep-Learning\\plant_Details.json", "r") as file:
                        plant_details = json.load(file)

                    for plant in detected_plant_names:
                        plant_info = next((p for p in plant_details["plants"] if p["Common Name"] == plant), None)
                        if plant_info:
                            st.write("Scientific Name:", plant_info["Scientific Name"])
                            st.write("Uses:", plant_info["Uses"]["Medicinal Uses"])
                            st.write("Location:", plant_info["Location"]["Native Region"])
                            st.write("Dosage:", plant_info["Dosage"]["Recommended Dosage"])
                            st.write("Active Compounds:", plant_info["Active Compounds"])
                            st.write("---")

                    # Automatically query the chatbot for information about the detected plants
                    for plant in detected_plant_names:
                        query = f"Tell me about {plant}."
                        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(query)
                        st.write(f"Chatbot Response for {plant}: {response.text}")

                else:
                    st.write("No plants detected.")

                # Provide a download link for the processed image
                with open(processed_image, "rb") as file:
                    st.download_button(label="Download Processed Image", data=file, file_name=f"processed_{filename}", mime="image/png")

                # Locate the TXT file with detection data
                txt_file = latest_run / f"{filename.split('.')[0]}.txt"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Read the existing detection data
                detection_data = ""

                if txt_file.exists():
                    with open(txt_file, "r") as file:
                        detection_data = file.read()

                # Append user information, purpose, detected plant names, and timestamp to the detection data
                detection_data += f"\nName: {user_name}\nAge: {user_age}\nPurpose: {selected_purpose}\nDetected Plants: {', '.join(detected_plant_names)}\nTimestamp: {timestamp}\n"
                detection_data += f"\n------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
                detection_data += f"Chatbot Response for {plant}: {response.text}\n\n"

                # Provide a download link for the updated detection data
                st.download_button(
                    label="Download Detection Data",
                    data=detection_data,
                    file_name=f"detection_{filename.split('.')[0]}.txt",
                    mime="text/plain"
                )
            else:
                st.write("No processed image available.")


    with st.expander("Select Your Below Choice", expanded=True):
        menu = st.selectbox('Menu', options=['Upload Image', 'Detect from webcam', 'Ask AI Chatbot', 'Detection History', 'Feedback', 'About Us'])
    if menu == 'Upload Image':
        # User inputs for name, age, and purpose of detection
        user_name = st.text_input("Enter your name:")
        user_age = st.text_input("Enter your age:")

        # Dropdown menu for detection purpose
        purpose_options = [
            "Identify Medicinal Properties",
            "Check Plant Authenticity",
            "Identify Edible Plants",
            "Identify Toxic Plants",
            "Determine Plant Dosage",
            "Educational Research",
            "Personal Use",
            "Commercial Use",
            "Others"
        ]
        selected_purpose = st.selectbox("Choose the purpose of detection:", purpose_options)

        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])

        if uploaded_file is not None:
            # Ensure upload folder exists
            if not os.path.exists('uploaded_images'):
                os.makedirs('uploaded_images')

            # Save the uploaded image
            filename = Path(uploaded_file.name).name
            image_path = os.path.join('uploaded_images', filename)

            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display the uploaded image
            st.image(image_path, caption='Uploaded Image', use_column_width=True)

            # Perform detection
            st.write("Processing the image...")
            model_path = "C:\\Users\\HP\\Downloads\\CSP_Medical-Plant-Detection-Using-Deep-Learning\\best.pt" # Path to your YOLO model
            model = YOLO(model_path)
            results = model.predict(image_path, save=True, save_txt=True)

            # Locate the processed image saved by YOLO
            runs_dir = Path("runs/detect")
            latest_run = max(runs_dir.iterdir(), key=os.path.getmtime)
            processed_image = latest_run / filename

            detected_plant_names = []

            if results:
                for result in results:
                    for box in result.boxes:
                        plant_name = result.names[int(box.cls)]
                        if plant_name not in detected_plant_names:  # Check if the plant name is already in the list
                            detected_plant_names.append(plant_name)
                            break

            if processed_image.exists():
                st.image(str(processed_image), caption='Processed Image', use_column_width=True)

                # Display detected plant names
                if detected_plant_names:
                    st.write("Detected Plants:")
                    for plant in detected_plant_names:
                        st.write(plant)

                    # Save detection history
                    detected_plants = ', '.join(detected_plant_names)
                    save_detection_history(user_name, user_age, selected_purpose, detected_plants)

                    # Load plant details from JSON
                    with open("C:\\Users\\HP\Downloads\\CSP_Medical-Plant-Detection-Using-Deep-Learning\\plant_Details.json", "r") as file:
                        plant_details = json.load(file)

                    for plant in detected_plant_names:
                        plant_info = next((p for p in plant_details["plants"] if p["Common Name"] == plant), None)
                        if plant_info:
                            st.write("*Scientific Name:*", plant_info["Scientific Name"])
                            st.write("*Uses:*", plant_info["Uses"]["Medicinal Uses"])
                            st.write("*Location:*", plant_info["Location"]["Native Region"])
                            st.write("*Dosage:*", plant_info["Dosage"]["Recommended Dosage"])
                            st.write("*Active Compounds:*", plant_info["Active Compounds"])
                            st.write("---")

                    # Automatically query the chatbot for information about the detected plants
                    for plant in detected_plant_names:
                        query = f"Tell me about {plant}."
                        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(query)
                        st.write(f"*Chatbot Response for {plant}:* {response.text}")

                else:
                    st.write("No plants detected.")

                # Provide a download link for the processed image
                with open(processed_image, "rb") as file:
                    st.download_button(label="Download Processed Image", data=file, file_name=f"processed_{filename}", mime="image/png")

                # Locate the TXT file with detection data
                txt_file = latest_run / f"{filename.split('.')[0]}.txt"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Read the existing detection data
                detection_data = ""

                if txt_file.exists():
                    with open(txt_file, "r") as file:
                        detection_data = file.read()

                # Append user information, purpose, detected plant names, and timestamp to the detection data
                detection_data += f"\nName: {user_name}\nAge: {user_age}\nPurpose: {selected_purpose}\nDetected Plants: {', '.join(detected_plant_names)}\nTimestamp: {timestamp}\n"
                detection_data += f"\n------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
                detection_data += f"Chatbot Response for {plant}: {response.text}\n\n"

                # Provide a download link for the updated detection data
                st.download_button(
                    label="Download Detection Data",
                    data=detection_data,
                    file_name=f"detection_{filename.split('.')[0]}.txt",
                    mime="text/plain"
                )
            else:
                st.write("No processed image available.")

    elif menu == 'Detect from webcam':
        st.write("Opening webcam for plant detection...")
        run_webcam_detection()

    elif menu == 'Ask AI Chatbot':
        st.subheader("Ask the Chatbot about Plants")
        user_query = st.text_input("What do you want to know about plants?", "")
        
        if st.button("Submit"):
            if user_query:
                # Generate a response from the Gemini API
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(user_query)
                st.write("Chatbot:", response.text)
            else:
                st.write("Please enter a question.")
    elif menu == 'Detection History':
            st.subheader("Detection History")
            detection_history = load_detection_history()
            if not detection_history.empty:
                st.table(detection_history)
            else:
                st.write("No detection history available.")
    elif menu == 'Feedback':
            st.subheader("Give Us Your Feedback")

            user_name = st.text_input("Enter your name:")
            user_age = st.text_input("Enter your age:")

            gender_options = ["Male", "Female", "Other"]
            selected_gender = st.selectbox("Select your gender:", gender_options)

            feedback_rating = st.radio("Rate your experience (1-5 stars):", range(1, 6))

            feedback_text = st.text_area("Share your suggestions:")

            if st.button("Submit Feedback"):
                save_feedback(user_name, user_age, selected_gender, feedback_rating, feedback_text)
                st.success("Thank you for your feedback!")
    elif menu == 'About Us':
        st.write("## About Us")
        st.write("We are a dedicated team committed to providing the best service 😀.")
        
        st.subheader("Our Mission:")
        st.write("1. Develop an accurate and reliable system for identifying medicinal plants using machine learning and computer vision techniques")
        st.write("2. Create a comprehensive database of Indian medicinal plants with detailed information on their properties, uses, and conservation status")
        st.write("3. Build a user-friendly web application to make medicinal plant identification accessible to botanists, researchers, and the general public")
        
        st.subheader("The Team:")
        st.write("Dr. [Name], Lead Researcher - Specializes in machine learning and image classification algorithms")
        st.write("[Name], Botanist - Provides domain expertise on medicinal plants and their identification")
        st.write("[Name], Web Developer - Responsible for designing and implementing the project's web application")
        st.write("[Name], Data Scientist - Analyzes plant data and develops insights to improve the identification system")
        
        st.subheader("Project Mentor:")
        st.write("ABC")
        
        st.subheader("Project Evaluator:")
        st.write("XYZ")
        
    else:
        st.write("Invalid selection.")
