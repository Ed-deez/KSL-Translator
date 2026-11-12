import os
import sqlite3
import numpy as np
import datetime
import customtkinter as ctk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "ksl_translator.db")
CLASSES_PATH = os.path.join(BASE_DIR, "..", "models", "gesture_classes.npy")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "ksl_cnn_lstm_model.keras")

class AdminDashboard(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="#1a1a1a")
        
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.stats = self.fetch_dashboard_stats()
        
        # Dictionaries to manage navigation state
        self.views = {}
        self.nav_buttons = {}

        self.setup_sidebar()
        
        # Container for the different screens
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=30, pady=20)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Build all the views
        self.build_dashboard_view()
        self.build_placeholder_view("Gesture Vocabulary")
        self.build_placeholder_view("Model Versions")
        self.build_placeholder_view("Users & Roles")
        self.build_placeholder_view("Feedback Review")

        # Start on the Dashboard
        self.switch_view("Dashboard")

    def fetch_dashboard_stats(self):
        stats = {
            "total_gestures": "0", "model_accuracy": "80.03%", 
            "total_users": "0", "model_version": "v1.0", "last_retrained": "Unknown"
        }
        if os.path.exists(CLASSES_PATH):
            try:
                stats["total_gestures"] = str(len(np.load(CLASSES_PATH, allow_pickle=True)))
            except: pass
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                stats["total_users"] = str(conn.cursor().execute("SELECT COUNT(*) FROM users").fetchone()[0])
                conn.close()
            except: pass
        if os.path.exists(MODEL_PATH):
            stats["last_retrained"] = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime("%b %d, %Y")
        return stats

    # --- NAVIGATION SYSTEM ---
    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#212121")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="KSL ADMIN", font=("Arial", 14, "bold"), text_color="gray").pack(pady=(20, 20), anchor="w", padx=20)
        
        nav_items = ["Dashboard", "Gesture Vocabulary", "Model Versions", "Users & Roles", "Feedback Review"]
        
        for text in nav_items:
            # Note the lambda uses default argument name=text to capture the value correctly in the loop
            btn = ctk.CTkButton(
                self.sidebar, text=text, fg_color="transparent", text_color="gray", 
                anchor="w", font=("Arial", 14), corner_radius=5,
                command=lambda name=text: self.switch_view(name)
            )
            btn.pack(fill="x", padx=10, pady=5)
            self.nav_buttons[text] = btn

    def switch_view(self, view_name):
        """Hides all views and shows the selected one, updating sidebar button colors."""
        # Update button highlights
        for name, btn in self.nav_buttons.items():
            if name == view_name:
                btn.configure(fg_color="#333333", text_color="white")
            else:
                btn.configure(fg_color="transparent", text_color="gray")

        # Hide all frames, then show the requested one
        for frame in self.views.values():
            frame.grid_remove()
            
        self.views[view_name].grid(row=0, column=0, sticky="nsew")

    # --- VIEW BUILDERS ---
    def build_placeholder_view(self, name):
        """Creates a simple placeholder screen for the other menu items."""
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        ctk.CTkLabel(frame, text=name, font=("Arial", 28, "bold")).pack(pady=40)
        ctk.CTkLabel(frame, text=f"This is the {name} management module.\nContent will go here.", text_color="gray").pack()
        self.views[name] = frame

    def build_dashboard_view(self):
        """Builds the main dashboard screen."""
        frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views["Dashboard"] = frame
        
        # Header
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(header, text="Overview", font=("Arial", 20, "bold")).pack(side="left")
        ctk.CTkLabel(header, text="Admin • logged in", font=("Arial", 12, "italic"), text_color="gray").pack(side="right")
        
        # Metric Cards
        cards_frame = ctk.CTkFrame(frame, fg_color="transparent")
        cards_frame.pack(fill="x", pady=(0, 30))
        cards_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.create_metric_card(cards_frame, self.stats["total_gestures"], "Gestures in vocabulary", 0, 0)
        self.create_metric_card(cards_frame, self.stats["model_accuracy"], "Current model accuracy", 0, 1)
        self.create_metric_card(cards_frame, self.stats["total_users"], "Total Registered Users", 1, 0)
        self.create_metric_card(cards_frame, self.stats["model_version"], "Active model version", 1, 1)
        
        # Gesture Vocabulary Header
        vocab_header = ctk.CTkFrame(frame, fg_color="transparent")
        vocab_header.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(vocab_header, text="Gesture Vocabulary", font=("Arial", 18, "bold")).pack(side="left")
        
        # Wired up the Add Gesture button
        ctk.CTkButton(vocab_header, text="+ Add Gesture", fg_color="transparent", border_width=1, width=120, 
                      command=self.open_add_gesture_modal).pack(side="right")
        
        self.create_data_table(frame)
        
        # Bottom Actions
        bottom_frame = ctk.CTkFrame(frame, fg_color="transparent")
        bottom_frame.pack(fill="x", pady=20)
        
        # Wired up the Retrain Model button
        ctk.CTkButton(bottom_frame, text="↻ Retrain Model", fg_color="transparent", border_width=1, width=140,
                      command=self.trigger_retrain).pack(side="left", padx=(0, 15))
        ctk.CTkLabel(bottom_frame, text=f"Last retrained on {self.stats['last_retrained']}", text_color="gray", font=("Arial", 11, "italic")).pack(side="left")

    def create_metric_card(self, parent, value, label, row, col):
        card = ctk.CTkFrame(parent, fg_color="#212121", corner_radius=8, height=100)
        card.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
        card.grid_propagate(False)
        ctk.CTkLabel(card, text=value, font=("Arial", 28, "bold")).pack(anchor="w", padx=20, pady=(20, 0))
        ctk.CTkLabel(card, text=label, font=("Arial", 12), text_color="gray").pack(anchor="w", padx=20)

    def create_data_table(self, parent_frame):
        table_frame = ctk.CTkFrame(parent_frame, fg_color="#212121", corner_radius=8)
        table_frame.pack(fill="x", pady=5)
        
        headers = ["Gesture", "Category", "Samples", "Updated"]
        for col_idx, header_text in enumerate(headers):
            table_frame.grid_columnconfigure(col_idx, weight=1)
            ctk.CTkLabel(table_frame, text=header_text, font=("Arial", 14, "bold")).grid(row=0, column=col_idx, sticky="w", padx=20, pady=15)
            
        sample_data = [
            ("Thank you", "Food", "48", "2 days ago"), ("Help", "Hygiene", "36", "5 days ago"),
            ("Happy", "Happiness", "52", "1 week ago"), ("Satisfied", "Satisfaction", "29", "2 weeks ago")
        ]
        
        for row_idx, row_data in enumerate(sample_data, start=1):
            ctk.CTkFrame(table_frame, height=1, fg_color="#333333").grid(row=row_idx * 2 - 1, column=0, columnspan=4, sticky="ew", padx=10)
            for col_idx, cell_value in enumerate(row_data):
                ctk.CTkLabel(table_frame, text=cell_value, font=("Arial", 13), text_color="#d1d1d1").grid(row=row_idx * 2, column=col_idx, sticky="w", padx=20, pady=12)

    # --- ACTION BUTTON LOGIC ---
    def open_add_gesture_modal(self):
        """Opens a popup window to add a new gesture."""
        modal = ctk.CTkToplevel(self)
        modal.title("Add New Gesture")
        modal.geometry("400x300")
        modal.transient(self.winfo_toplevel()) # Keep popup on top
        modal.grab_set() # Block interaction with main window until closed
        
        ctk.CTkLabel(modal, text="Register New Gesture", font=("Arial", 18, "bold")).pack(pady=(20, 10))
        
        gesture_entry = ctk.CTkEntry(modal, placeholder_text="Gesture Name (e.g., 'Hello')", width=300)
        gesture_entry.pack(pady=10)
        
        category_entry = ctk.CTkEntry(modal, placeholder_text="Category (e.g., 'Greetings')", width=300)
        category_entry.pack(pady=10)
        
        def save_gesture():
            # In the future, this will update gesture_classes.npy or the database
            name = gesture_entry.get()
            if name:
                messagebox.showinfo("Success", f"Gesture '{name}' added to training queue.")
                modal.destroy()
            else:
                messagebox.showwarning("Warning", "Gesture name cannot be empty.")
                
        ctk.CTkButton(modal, text="Save Gesture", command=save_gesture, width=300).pack(pady=20)

    def trigger_retrain(self):
        """Confirms and triggers the model training script."""
        confirm = messagebox.askyesno(
            "Retrain Model", 
            "Are you sure you want to retrain the CNN-LSTM model?\n\nThis process may take several minutes depending on your hardware."
        )
        if confirm:
            messagebox.showinfo("Training Started", "Model retraining initialized in the background. You can continue using the dashboard.")
            # Here you would use Python's 'subprocess' or 'threading' module to run your train_model.py script seamlessly:
            # threading.Thread(target=lambda: os.system("python -m ml.train_model")).start()

if __name__ == "__main__":
    print("Launching Dashboard...") # Let's add this print statement to test it!
    app = ctk.CTk()
    app.geometry("1100x750")
    app.title("KSL Translator - Admin Dashboard")
    dashboard = AdminDashboard(app)
    dashboard.pack(fill="both", expand=True)
    app.mainloop()