import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp

class TranslatePage(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Initialize standard MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2 # Adjust to 1 if your specific KSL dataset only uses one hand
        )
        
        self.setup_camera_feed()
        self.setup_sidebar()
        
        # Start camera (0 is the default webcam)
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def setup_camera_feed(self):
        self.camera_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=10)
        self.camera_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        self.video_label = ctk.CTkLabel(self.camera_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.cam_controls = ctk.CTkFrame(self.camera_frame, fg_color="transparent")
        self.cam_controls.pack(fill="x", pady=10, padx=20)
        
        self.fullscreen_btn = ctk.CTkButton(self.cam_controls, text="Fullscreen", width=100, fg_color="transparent", border_width=1)
        self.fullscreen_btn.pack(side="left", padx=5)
        
        self.settings_btn = ctk.CTkButton(self.cam_controls, text="Camera Settings", width=120, fg_color="transparent", border_width=1)
        self.settings_btn.pack(side="left", padx=5)

    def setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Detecting gesture...", text_color="gray")
        self.status_label.pack(anchor="w", pady=(0, 10))
        
        self.text_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#2b2b2b", corner_radius=5)
        self.text_frame.pack(fill="x", pady=20)
        
        ctk.CTkLabel(self.text_frame, text="RECOGNIZED TEXT", text_color="gray", font=("Arial", 10)).pack(anchor="w", padx=15, pady=(10, 0))
        self.translation_output = ctk.CTkLabel(self.text_frame, text='"..."', font=("Arial", 20, "bold"))
        self.translation_output.pack(anchor="w", padx=15, pady=(5, 15))
        
        self.play_btn = ctk.CTkButton(self.sidebar_frame, text="Play speech", fg_color="white", text_color="black")
        self.play_btn.pack(fill="x", pady=(0, 10))
        
        self.repeat_btn = ctk.CTkButton(self.sidebar_frame, text="Repeat gesture", fg_color="transparent", border_width=1)
        self.repeat_btn.pack(fill="x", pady=(0, 30))

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip horizontally for a natural selfie-view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR (OpenCV) to RGB (MediaPipe and Pillow)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to detect hands
            results = self.hands.process(rgb_frame)
            
            # Draw the landmarks and connections
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        rgb_frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
                    
            # Convert the annotated RGB image to a Tkinter-compatible format
            img = Image.fromarray(rgb_frame)
            img = img.resize((600, 400), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Loop the camera update every 15 milliseconds
        self.after(15, self.update_camera)

    def destroy(self):
        # Release hardware resources to prevent memory leaks or crashes
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        super().destroy()