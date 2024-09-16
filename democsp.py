#%%writefile CSP_Medplant.py
import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import bcrypt
import base64
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import re
import json
import google.generativeai as genai
#import cv2
import numpy as np
import time
from fpdf import FPDF
# Paths for user data
user_data_file = "login_data.csv"
detection_history_file = "Detection History.csv"
feedback_file = "feedback.csv"


#api_key = st.secrets["api_key"]

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
    rating_map = {
        1: "1 Star - Poor",
        2: "2 Stars - Fair",
        3: "3 Stars - Average",
        4: "4 Stars - Good",
        5: "5 Stars - Excellent"
    }
    formatted_rating = rating_map[rating]
    
    
    feedback_data = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Gender": [gender],
        "Rating": [formatted_rating],
        "Feedback": [feedback]
    })

    if os.path.exists(feedback_file):
        existing_data = pd.read_csv(feedback_file)
        feedback_data = pd.concat([existing_data, feedback_data], ignore_index=True)

    feedback_data.to_csv(feedback_file, index=False)

# Initialize session state for detection history
if 'history_saved' not in st.session_state:
    st.session_state.history_saved = False
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
        menu = option_menu(
                            menu_title=None,
                            options=['Login', 'Register', 'Forgot Password'],
                            icons=['box-arrow-right', 'person-plus', 'key'],
                            orientation='horizontal'
  
                          )

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
            st.subheader('Reset Password')
            username = st.text_input("Enter your Username")
            new_password = st.text_input("Enter your New Password", type='password')
            confirm_password = st.text_input("Confirm New Password", type='password')

            if st.button("Reset Password"):
                if username_exists(username):
                    if new_password == confirm_password:
                        if change_password(username, new_password):
                            st.success("Your password has been reset successfully.")
                        else:
                            st.error("Failed to reset password. Please try again.")
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
                    st.session_state.show_success_message = True  # Flag to show success message
                    st.rerun()
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
    with open("medplant bg.jpg", "rb") as image_file:  # Change this path to your image
        encoded_image = base64.b64encode(image_file.read()).decode()

    # Set the background
    set_background(encoded_image)
    
    # Display the "Login successful!" message if the user has just logged in
    if 'show_success_message' in st.session_state and st.session_state.show_success_message:
        st.success("Login successful!✅")
        time.sleep(3)  # Keep the message for 5 seconds
        st.session_state.show_success_message = False
        st.rerun() # Reset the flag
    # Main title for the project features
    st.title("Medical Plant Image Detection")

    def run_webcam_detection():
        # User inputs for name, age, and purpose of detection
        user_name = st.text_input("Enter Your Name:")
        user_age = st.number_input("Enter Your Age:", min_value=1, max_value=120, step=1, format="%d")
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
        selected_purpose = st.selectbox("Choose The Purpose of Detection:", purpose_options)

        camera = st.camera_input("Capture an Image from Your Webcam")

        if camera is not None:
            # Ensure upload folder exists
            if not os.path.exists('uploaded_images'):
                os.makedirs('uploaded_images')
            if user_name and user_age:
                # Save the uploaded image
                filename = Path(camera.name).name
                image_path = os.path.join('uploaded_images', filename)

                with open(image_path, "wb") as f:
                    f.write(camera.getbuffer())

                # Perform detection
                st.write("Processing the image...")
                model_path = "best.pt"  # Path to your YOLO model
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
                        with open("plant_Details.json", "r") as file:
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
                        st.download_button(label="Download Processed Image as .jpg", data=file, file_name=f"processed_{filename}", mime="image/png")

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
                        label="Download Detection Data as .txt",
                        data=detection_data,
                        file_name=f"detection_{filename.split('.')[0]}.txt",
                        mime="text/plain"
                      
                    )
                    # Function to create a PDF with tabular formatting
                    def create_detection_pdf():
                        pdf = FPDF()
                        pdf.add_page()

                        # Set fonts and colors
                        pdf.set_font('Arial', 'B', 16)
                        pdf.set_fill_color(255, 255, 255)  # Background color (white)
                        
                        # Table title
                        title = 'Medical Plant Detection Report'
                        pdf.cell(0, 10, title, ln=True, align='C')
                        pdf.set_font('Arial', '', 12)

                        # Move down slightly to avoid overlap with the underline
                        pdf.ln(10)
                
                        # Define a function to add table rows
                        def add_table_row(label, value):
                            pdf.cell(90, 10, label, border=1, align='L')
                            pdf.cell(0, 10, value, border=1, ln=True, align='L')
                        # Add data rows
                        add_table_row("Name", user_name)
                        add_table_row("Age", str(user_age))
                        add_table_row("Purpose", selected_purpose)
                        add_table_row("Detected Plants", ', '.join(detected_plant_names))
                        add_table_row("Timestamp", timestamp)

                        # Add separator line
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, '-' * 140, ln=True)

                        # Title for the processed image
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, 'Processed Image', ln=True, align='L')
                        # Add image
                        pdf.ln(10)
                        if processed_image.exists():
                            pdf.image(str(processed_image), x=10, y=pdf.get_y(), w=180)  # Adjust x, y, and w as needed
                        
                        pdf.ln(100)
                        # Title for the processed image
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, 'AI Chatbot Response', ln=True, align='L')

                        
                        # Chatbot response
                        pdf.set_font('Arial', '', 12)

                        # Split the response text into lines and add them with bold formatting where needed
                        response_lines = response.text.splitlines()
                        for line in response_lines:
                            if '**' in line:  # Markdown-like bold formatting detection
                                # Split the line around '**' to separate bold and normal text
                                parts = line.split('**')
                                for i, part in enumerate(parts):
                                    if i % 2 == 1:  # Bold text
                                        pdf.set_font('Arial', 'B', 12)
                                    else:  # Regular text
                                        pdf.set_font('Arial', '', 12)
                                    pdf.multi_cell(0, 10, part, align='L')
                            else:
                                pdf.set_font('Arial', '', 12)
                                pdf.multi_cell(0, 10, line, align='L')

                        # Return PDF as a binary string
                        return pdf.output(dest='S').encode('latin1')  # Return PDF as a binary string
                    # Create and provide the download link for PDF
                    pdf_data = create_detection_pdf()

                    st.download_button(
                        label="Download Detection Data as .pdf",
                        data=pdf_data,
                        file_name=f"detection_{filename.split('.')[0]}.pdf",
                        mime="application/pdf"
                                        )

                else:
                    st.write("No processed image available.")
            else:
                st.error("Please fill in all fields before Capturing an image!")

    # Function to create a PDF with the chatbot response
    def create_response_pdf(response_text):
        pdf = FPDF()
        pdf.add_page()

        # Set fonts and colors
        pdf.set_font('Arial', 'B', 16)
        pdf.set_fill_color(255, 255, 255)  # Background color (white)
        
        # Title
        title = 'Response for Your Query'
        pdf.cell(0, 10, title, ln=True, align='C')

        # Add the response text
        pdf.ln(10)
        pdf.set_font('Arial', '', 12)
        
        # Replace bold markers with FPDF's bold
        formatted_text = response_text.replace('*', '')
        pdf.multi_cell(0, 10, formatted_text)
        
        return pdf.output(dest='S').encode('latin1')  # Return PDF as a binary string

        
    with st.sidebar:
        selected=option_menu(
            menu_title='Main Menu',
            options=['Upload Image', 'Detect from webcam', 'Ask AI Chatbot', 'Detection History', 'Feedback', 'About Us', 'Logout'],
            icons=['upload', 'webcam-fill', 'robot', 'hourglass-bottom', 'star-fill', 'file-earmark-person', 'box-arrow-left'],
            menu_icon='cast',
            default_index=0
        )
    if selected == 'Upload Image':
        # User inputs for name, age, and purpose of detection
        user_name = st.text_input("Enter Your Name:")
        user_age = st.number_input("Enter Your Age:", min_value=1, max_value=120, step=1, format="%d")

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
        selected_purpose = st.selectbox("Choose the Purpose of Detection:", purpose_options)

        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])

        if uploaded_file is not None:
            # Ensure upload folder exists
            if not os.path.exists('uploaded_images'):
                os.makedirs('uploaded_images')
            if user_name and user_age and selected_purpose:
                
                # Save the uploaded image
                filename = Path(uploaded_file.name).name
                image_path = os.path.join('uploaded_images', filename)

                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Display the uploaded image
                st.image(image_path, caption='Uploaded Image', use_column_width=True)

                # Perform detection
                st.write("Processing the image...")
                model_path = "best.pt" # Path to your YOLO model
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
                        with open("plant_Details.json", "r") as file:
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
                        st.download_button(label="Download Processed Image as .jpg", data=file, file_name=f"processed_{filename}", mime="image/png")

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
                        label="Download Detection Data as .txt",
                        data=detection_data,
                        file_name=f"detection_{filename.split('.')[0]}.txt",
                        mime="text/plain"
                    )

                    # Function to create a PDF with tabular formatting
                    def create_detection_pdf():
                        pdf = FPDF()
                        pdf.add_page()

                        # Set fonts and colors
                        pdf.set_font('Arial', 'B', 16)
                        pdf.set_fill_color(255, 255, 255)  # Background color (white)
                        
                        # Table title
                        title = 'Medical Plant Detection Report'
                        pdf.cell(0, 10, title, ln=True, align='C')
                        pdf.set_font('Arial', '', 12)

                        # Move down slightly to avoid overlap with the underline
                        pdf.ln(10)
                
                        # Define a function to add table rows
                        def add_table_row(label, value):
                            pdf.cell(90, 10, label, border=1, align='L')
                            pdf.cell(0, 10, value, border=1, ln=True, align='L')
                        # Add data rows
                        add_table_row("Name", user_name)
                        add_table_row("Age", str(user_age))
                        add_table_row("Purpose", selected_purpose)
                        add_table_row("Detected Plants", ', '.join(detected_plant_names))
                        add_table_row("Timestamp", timestamp)

                        # Add separator line
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, '-' * 140, ln=True)

                        # Title for the processed image
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, 'Processed Image', ln=True, align='L')
                        # Add image
                        pdf.ln(10)
                        if processed_image.exists():
                            pdf.image(str(processed_image), x=10, y=pdf.get_y(), w=180)  # Adjust x, y, and w as needed
                        
                        pdf.ln(100)
                        # Title for the processed image
                        pdf.ln(10)
                        pdf.set_font('Arial', 'B', 12)
                        pdf.cell(0, 10, 'AI Chatbot Response', ln=True, align='L')

                        
                        # Chatbot response
                        pdf.set_font('Arial', '', 12)

                        # Split the response text into lines and add them with bold formatting where needed
                        response_lines = response.text.splitlines()
                        for line in response_lines:
                            if '**' in line:  # Markdown-like bold formatting detection
                                # Split the line around '**' to separate bold and normal text
                                parts = line.split('**')
                                for i, part in enumerate(parts):
                                    if i % 2 == 1:  # Bold text
                                        pdf.set_font('Arial', 'B', 12)
                                    else:  # Regular text
                                        pdf.set_font('Arial', '', 12)
                                    pdf.multi_cell(0, 10, part, align='L')
                            else:
                                pdf.set_font('Arial', '', 12)
                                pdf.multi_cell(0, 10, line, align='L')

                        # Return PDF as a binary string
                        return pdf.output(dest='S').encode('latin1')  # Return PDF as a binary string
                    # Create and provide the download link for PDF
                    pdf_data = create_detection_pdf()

                    st.download_button(
                        label="Download Detection Data as .pdf",
                        data=pdf_data,
                        file_name=f"detection_{filename.split('.')[0]}.pdf",
                        mime="application/pdf"
                                        )

                else:
                    st.write("No processed image available.")
            else:
                st.error("Please fill in all fields before uploading your image!")
    if selected == 'Detect from webcam':
        st.write("Opening webcam for plant detection...")
        run_webcam_detection()

    if selected == 'Ask AI Chatbot':
        st.subheader("Ask the Chatbot about Plants")
        user_query = st.text_input("What do you want to know about plants?",
                                 placeholder="Ask your Medical Plants related question here...")
        
        if st.button("Submit"):
            if user_query:
                # Generate a response from the Gemini API
                response = genai.GenerativeModel('gemini-1.5-flash').generate_content(user_query)
                st.write("Chatbot:", response.text)
                # Provide an option to download the response as PDF
                pdf_data = create_response_pdf(response.text)
                st.download_button(
                    label="Download Response as PDF",
                    data=pdf_data,
                    file_name="chatbot_response.pdf",
                    mime="application/pdf"
                                  )
            else:
                st.warning("⚠️ Please ask your question.")  # Show a warning if input is empty
    if selected == 'Detection History':
            st.subheader("Detection History")
            detection_history = load_detection_history()
            if not detection_history.empty:
                st.table(detection_history)
            else:
                st.write("No detection history available.")
    if selected == 'Feedback':
            st.subheader("Give Us Your Feedback")

            user_name = st.text_input("Enter Your Name:")
            user_age = st.number_input("Enter Your Age:", min_value=1, max_value=120, step=1, format="%d")

            gender_options = ["Male", "Female"]
            selected_gender = st.selectbox("Select Your Gender:", gender_options)

            #feedback_rating = st.radio("Rate your experience (1-5 stars):", range(1, 6))
            # Rating selection
            feedback_rating = st.radio(
                "Rate Your Experience (1-5 stars):",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: {
                    1: "1 Star - Poor",
                    2: "2 Stars - Fair",
                    3: "3 Stars - Average",
                    4: "4 Stars - Good",
                    5: "5 Stars - Excellent"
                }[x]
            )

            feedback_text = st.text_area("Share Your Suggestions: (if any)")

            if st.button("Submit Feedback"):
                if user_name and user_age and selected_gender:
                    
                    save_feedback(user_name, user_age, selected_gender, feedback_rating, feedback_text)
                    st.success("Thank you for your feedback!")
                else:
                    st.error("Please fill in all fields before submitting.")
    if selected == 'About Us':
                # Add Font Awesome CDN link to your Streamlit app
        st.markdown(
            """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
            """,
            unsafe_allow_html=True
        )
        st.markdown("## <i class='fas fa-info-circle'></i> About Us", unsafe_allow_html=True)
        st.write("We are a dedicated team committed to providing the best service.")

        # Mission Section
        st.markdown("<h3><i class='fas fa-bullseye'></i> Our Mission:</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. <i class='fas fa-seedling'></i> **Develop an accurate and reliable system** for identifying medicinal plants using machine learning and computer vision techniques.
        2. <i class='fas fa-book'></i> **Create a comprehensive database** of Indian medicinal plants with detailed information on their properties, uses, and conservation status.
        3. <i class='fas fa-laptop-code'></i> **Build a user-friendly web application** to make medicinal plant identification accessible to botanists, researchers, and the general public.
        """, unsafe_allow_html=True)

        # Team Section
        st.markdown("<h3><i class='fas fa-users'></i> The Team:</h3>", unsafe_allow_html=True)
        st.markdown("""
        - <i class='fas fa-user'></i> **Dr. [Name], Lead Researcher** - Specializes in machine learning and image classification algorithms.
        - <i class='fas fa-user'></i> **[Name], Botanist** - Provides domain expertise on medicinal plants and their identification.
        - <i class='fas fa-user'></i> **[Name], Web Developer** - Responsible for designing and implementing the project's web application.
        - <i class='fas fa-user'></i> **[Name], Data Scientist** - Analyzes plant data and develops insights to improve the identification system.
        """, unsafe_allow_html=True)

        # Project Mentor
        st.markdown("<h3><i class='fas fa-chalkboard-teacher'></i> Project Mentor:</h3>", unsafe_allow_html=True)
        st.write("**ABC**")

        # Project Evaluator
        st.markdown("<h3><i class='fas fa-pen'></i> Project Evaluator:</h3>", unsafe_allow_html=True)
        st.write("**XYZ**")

        # Adding some separation
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3><i class='fas fa-star'></i> We are committed to using science and technology to benefit the community. This project is a part of our community service efforts, aimed at improving medicinal plant identification and making this valuable information accessible to all !!</h3>", unsafe_allow_html=True)

    # Logout button
    if selected == 'Logout':
        st.session_state.logged_in = False
        st.session_state.logout_message = "You have successfully logged out! Please log in again to continue your exploration of medicinal plant detection."
        st.rerun()  # Refresh the page
