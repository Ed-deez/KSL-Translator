import os
import cv2
import numpy as np
import mediapipe as mp
import customtkinter as ctk
from PIL import Image
from tensorflow.keras.models import load_model
from tkinter import messagebox
import pyttsx3
import threading

try:
    from database import database
    from ui.admin_dashboard import AdminDashboard
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from database import database
    from ui.admin_dashboard import AdminDashboard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ksl_cnn_lstm_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "..", "models", "gesture_classes.npy")

class MainWindow(ctk.CTk):
    def __init__(self, current_user_email="Guest"):
        super().__init__()
        
        self.current_user_email = current_user_email
        self.title("KSL Translator - Loading...")
        self.geometry("1150x750")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- 1. State Variables ---
        self.sentence = []
        self.current_gesture = None
        self.gesture_counter = 0
        self.required_frames = 15
        self.last_added_gesture = None
        
        self.camera_paused = True # Keep paused during loading
        self.settings_open = False
        self.current_theme = "dark"

        # --- 2. Setup Loading Screen ---
        self.loading_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.loading_frame.pack(fill="both", expand=True)
        
        # Loading Logo / Icon
        self.logo_label = ctk.CTkLabel(self.loading_frame, text="👋", font=("Arial", 80))
        self.logo_label.pack(pady=(200, 20))

        self.loading_title = ctk.CTkLabel(self.loading_frame, text="KSL Translator", font=("Arial", 32, "bold"))
        self.loading_title.pack(pady=10)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.loading_frame, width=400, height=15, corner_radius=10)
        self.progress_bar.pack(pady=20)
        self.progress_bar.set(0.0) # Start at 0%

        self.status_label = ctk.CTkLabel(self.loading_frame, text="Initializing environment...", text_color="gray", font=("Arial", 14))
        self.status_label.pack()

        # Start the loading sequence after 200ms so the UI has time to render the loading screen
        self.after(200, self.load_step_1_mediapipe)

    # --- 3. Staggered Loading Sequence ---
    def load_step_1_mediapipe(self):
        self.status_label.configure(text="Loading MediaPipe Vision tracking...")
        self.progress_bar.set(0.25)
        self.update_idletasks() # Force UI to visually update
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.after(100, self.load_step_2_tensorflow)

    def load_step_2_tensorflow(self):
        self.status_label.configure(text="Loading CNN-LSTM Neural Network (This may take a moment)...")
        self.progress_bar.set(0.60)
        self.update_idletasks()
        
        self.model = load_model(MODEL_PATH)
        self.classes = np.load(CLASSES_PATH, allow_pickle=True)
        
        self.after(100, self.load_step_3_camera)

    def load_step_3_camera(self):
        self.status_label.configure(text="Connecting to Webcam...")
        self.progress_bar.set(0.85)
        self.update_idletasks()
        
        self.cap = cv2.VideoCapture(0)
        
        self.after(100, self.load_step_4_ui)

    def load_step_4_ui(self):
        self.status_label.configure(text="Building Interface...")
        self.progress_bar.set(1.0)
        self.update_idletasks()
        
        # Destroy the loading screen
        self.loading_frame.pack_forget()
        self.loading_frame.destroy()
        self.title("KSL Translator - Live Feed")
        
        # Build the actual app
        self.build_main_ui()
        
        # Unpause camera and start video loop
        self.camera_paused = False
        self.update_frame()

    # --- 4. Build Main UI (Moved from __init__) ---
    def build_main_ui(self):
        # Sidebar Setup
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y")
        self.sidebar_frame.pack_propagate(False)
        
        self.settings_btn = ctk.CTkButton(
            self.sidebar_frame, text="⚙  Settings  ▾", font=("Arial", 16, "bold"),
            fg_color="transparent", hover_color="#333333", anchor="w", command=self.toggle_settings_menu
        )
        self.settings_btn.pack(fill="x", padx=15, pady=(20, 5))

        self.settings_menu_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#242424", corner_radius=8)
        
        self.nav_translator_btn = ctk.CTkButton(self.settings_menu_frame, text="📹  Live Translator", fg_color="#333333", hover_color="#444444", anchor="w", command=self.show_translator_view)
        self.nav_translator_btn.pack(fill="x", padx=8, pady=(8, 4))

        self.nav_admin_btn = ctk.CTkButton(self.settings_menu_frame, text="📊  Admin Dashboard", fg_color="transparent", hover_color="#444444", anchor="w", command=self.show_admin_view)
        self.nav_admin_btn.pack(fill="x", padx=8, pady=4)

        self.theme_btn = ctk.CTkButton(self.settings_menu_frame, text="🌓  Switch Theme", fg_color="transparent", hover_color="#444444", anchor="w", command=self.toggle_theme)
        self.theme_btn.pack(fill="x", padx=8, pady=(4, 8))

        self.logout_btn = ctk.CTkButton(self.sidebar_frame, text="Log Out", command=self.logout)
        self.logout_btn.pack(side="bottom", pady=15, padx=20, fill="x")
        
        self.delete_btn = ctk.CTkButton(self.sidebar_frame, text="Delete Account", fg_color="#8B0000", hover_color="#FF0000", command=self.delete_account)
        self.delete_btn.pack(side="bottom", pady=(10, 0), padx=20, fill="x")

        # Main Translation Frame
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.title_label = ctk.CTkLabel(self.main_frame, text="Real-Time Translation", font=("Arial", 28, "bold"))
        self.title_label.pack(pady=5)

        self.sentence_textbox = ctk.CTkTextbox(self.main_frame, height=80, font=("Arial", 28), fg_color="#1a1a1a", text_color="#00FF00")
        self.sentence_textbox.pack(pady=10, fill="x", padx=20)
        self.sentence_textbox.insert("0.0", "Waiting for gesture...")
        self.sentence_textbox.configure(state="disabled") 
        
        self.actions_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.actions_frame.pack(pady=5)
        
        self.clear_btn = ctk.CTkButton(self.actions_frame, text="Clear Text", command=self.clear_sentence)
        self.clear_btn.pack(side="left", padx=10)

        self.speak_btn = ctk.CTkButton(self.actions_frame, text="Speak Sentence", fg_color="#2b5b84", command=self.speak_sentence)
        self.speak_btn.pack(side="left", padx=10)

        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(pady=10, expand=True)

        self.admin_view = AdminDashboard(self)

    # --- 5. Settings Menu & View Switching ---
    def toggle_settings_menu(self):
        if self.settings_open:
            self.settings_menu_frame.pack_forget()
            self.settings_btn.configure(text="⚙  Settings  ▾")
            self.settings_open = False
        else:
            self.settings_menu_frame.pack(fill="x", padx=15, pady=(0, 10), after=self.settings_btn)
            self.settings_btn.configure(text="⚙  Settings  ▴")
            self.settings_open = True

    def show_admin_view(self):
        self.camera_paused = True
        self.main_frame.pack_forget()
        self.admin_view.pack(side="right", fill="both", expand=True)
        self.nav_admin_btn.configure(fg_color="#333333")
        self.nav_translator_btn.configure(fg_color="transparent")

    def show_translator_view(self):
        self.admin_view.pack_forget()
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        self.camera_paused = False
        self.nav_translator_btn.configure(fg_color="#333333")
        self.nav_admin_btn.configure(fg_color="transparent")

    def toggle_theme(self):
        if self.current_theme == "dark":
            ctk.set_appearance_mode("light")
            self.current_theme = "light"
        else:
            ctk.set_appearance_mode("dark")
            self.current_theme = "dark"

    # --- 6. Processing Loop ---
    def update_frame(self):
        if not self.camera_paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                hand_detected = False
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        coords = []
                        for landmark in hand_landmarks.landmark:
                            coords.extend([landmark.x, landmark.y, landmark.z])
                            
                        if len(coords) == 63:
                            hand_detected = True
                            input_data = np.array(coords).reshape(1, 1, 63)
                            prediction = self.model.predict(input_data, verbose=0)
                            class_idx = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            if confidence > 0.60: 
                                detected_gesture = self.classes[class_idx]
                                if detected_gesture == self.current_gesture:
                                    self.gesture_counter += 1
                                else:
                                    self.current_gesture = detected_gesture
                                    self.gesture_counter = 1
                                    
                                if self.gesture_counter >= self.required_frames and detected_gesture != self.last_added_gesture:
                                    self.sentence.append(detected_gesture)
                                    self.last_added_gesture = detected_gesture
                                    self.update_sentence_ui()
                
                if not hand_detected:
                    self.current_gesture = None
                    self.gesture_counter = 0
                    self.last_added_gesture = None

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

        self.after(10, self.update_frame)

    # --- 7. Helper Functions & Text-to-Speech ---
    def update_sentence_ui(self):
        self.sentence_textbox.configure(state="normal")
        self.sentence_textbox.delete("0.0", "end")
        if self.sentence:
            self.sentence_textbox.insert("0.0", " ".join(self.sentence))
        else:
            self.sentence_textbox.insert("0.0", "Waiting for gesture...")
        self.sentence_textbox.configure(state="disabled")

    def clear_sentence(self):
        self.sentence = []
        self.last_added_gesture = None
        self.update_sentence_ui()

    def speak_sentence(self):
        if not self.sentence: return
        threading.Thread(target=self._run_tts, args=(" ".join(self.sentence),)).start()

    def _run_tts(self, text):
        engine = pyttsx3.init()
        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate - 25) 
        engine.say(text)
        engine.runAndWait()

    # --- 8. Navigation & Security ---
    def logout(self):
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        self.destroy()
        try: from ui.login_window import LoginWindow
        except ImportError: from login_window import LoginWindow
        login_app = LoginWindow()
        login_app.mainloop()

    def delete_account(self):
        if self.current_user_email == "Guest":
            messagebox.showinfo("Info", "Guest accounts cannot be deleted.")
            return
        if messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete your account? This cannot be undone."):
            if database.delete_user(self.current_user_email):
                messagebox.showinfo("Deleted", "Your account has been successfully deleted.")
                self.logout()
            else:
                messagebox.showerror("Error", "Failed to delete account.")

    def on_closing(self):
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()