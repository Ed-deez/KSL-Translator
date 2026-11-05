import os
import cv2
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from PIL import Image
from tensorflow.keras.models import load_model
from tkinter import messagebox
from database import database # Adjust this import if your DB file is in a different folder

# Dynamically set paths to load the brain
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ksl_cnn_lstm_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "..", "models", "gesture_classes.npy")

class MainWindow(ctk.CTk):
    def __init__(self, current_user_email="Guest"):
        super().__init__()
        
        self.current_user_email = current_user_email
        self.title("KSL Translator - Live Feed")
        self.geometry("1000x700")
        
        # Ensure the camera releases if the user clicks the red 'X' to close the window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- 1. Load Machine Learning Models ---
        print("Loading CNN-LSTM Model...")
        self.model = load_model(MODEL_PATH)
        self.classes = np.load(CLASSES_PATH, allow_pickle=True)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, # False because this is a live video stream
            max_num_hands=2, 
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- 2. Setup the UI ---
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")
        
        ctk.CTkLabel(self.sidebar_frame, text="Settings", font=("Arial", 20, "bold")).pack(pady=20)
        
        self.logout_btn = ctk.CTkButton(self.sidebar_frame, text="Log Out", command=self.logout)
        self.logout_btn.pack(side="bottom", pady=10, padx=20)
        
        self.delete_btn = ctk.CTkButton(self.sidebar_frame, text="Delete Account", fg_color="#8B0000", hover_color="#FF0000", command=self.delete_account)
        self.delete_btn.pack(side="bottom", pady=(10, 0), padx=20)

        # Main Content Area
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(self.main_frame, text="Real-Time Translation", font=("Arial", 28, "bold"))
        self.title_label.pack(pady=10)

        # Output Translation Label
        self.translation_var = ctk.StringVar(value="Waiting for gesture...")
        self.translation_label = ctk.CTkLabel(self.main_frame, textvariable=self.translation_var, font=("Arial", 36, "bold"), text_color="#00FF00")
        self.translation_label.pack(pady=10)

        # Video Feed Label
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(pady=10, expand=True)

        # --- 3. Start Camera and Processing Loop ---
        self.cap = cv2.VideoCapture(0) # 0 is usually the default built-in webcam
        self.update_frame()

    def update_frame(self):
        """This function loops every 10 milliseconds to process the live video feed."""
        ret, frame = self.cap.read()
        
        if ret:
            # Flip frame to act like a mirror and convert colors for MediaPipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the tracking lines on the screen
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract the 63 Coordinates
                    coords = []
                    for landmark in hand_landmarks.landmark:
                        coords.extend([landmark.x, landmark.y, landmark.z])
                        
                    if len(coords) == 63:
                        # Format the data for the CNN-LSTM (samples, time_steps, features)
                        input_data = np.array(coords).reshape(1, 1, 63)
                        
                        # Make the prediction
                        prediction = self.model.predict(input_data, verbose=0)
                        class_idx = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        # Only update the UI if the model is at least 60% confident
                        if confidence > 0.60: 
                            detected_gesture = self.classes[class_idx]
                            self.translation_var.set(f"Translation: {detected_gesture}")

            # Convert the OpenCV frame into a CustomTkinter Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            
            # Update the label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Schedule this function to run again in 10 milliseconds
        self.after(10, self.update_frame)

    # --- CRUD and Navigation Functions ---
    def logout(self):
        self.cap.release() # Turn off the webcam
        self.destroy()
        
        # Adjust the import based on your exact login script name
        from login_window import LoginWindow 
        login_app = LoginWindow()
        login_app.mainloop()

    def delete_account(self):
        if self.current_user_email == "Guest":
            messagebox.showinfo("Info", "Guest accounts cannot be deleted.")
            return
            
        confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete your account? This cannot be undone.")
        
        if confirm:
            success = database.delete_user(self.current_user_email)
            if success:
                messagebox.showinfo("Deleted", "Your account has been successfully deleted.")
                self.logout()
            else:
                messagebox.showerror("Error", "Failed to delete account.")

    def on_closing(self):
        """Safely release the camera when clicking the window's close button."""
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()