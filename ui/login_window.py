import os
import sys
from tkinter import messagebox  # For showing error/success popups

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import database
import customtkinter as ctk

class RegisterWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("KSL Translator - Create Account")
        self.geometry("400x550")
        self.configure(fg_color="#1a1a1a")
        
        # Make the popup grab focus
        self.transient(parent)
        self.grab_set()
        
        # Default role
        self.selected_role = ctk.StringVar(value="Signer")
        
        self.setup_ui()

    def setup_ui(self):
        # Header
        self.header_label = ctk.CTkLabel(self, text="Create an account", font=("Arial", 20, "bold"))
        self.header_label.pack(pady=(30, 20))
        
        # Role Selection
        self.role_label = ctk.CTkLabel(self, text="Select your role:", text_color="gray")
        self.role_label.pack(anchor="w", padx=40)
        
        self.role_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.role_frame.pack(fill="x", padx=40, pady=(5, 15))
        
        self.radio_signer = ctk.CTkRadioButton(self.role_frame, text="Signer", variable=self.selected_role, value="Signer")
        self.radio_signer.pack(side="left", padx=(0, 20))
        
        self.radio_hearing = ctk.CTkRadioButton(self.role_frame, text="Hearing User", variable=self.selected_role, value="Hearing User")
        self.radio_hearing.pack(side="left")
        
        # Email Input
        self.email_entry = ctk.CTkEntry(self, placeholder_text="name@example.com", width=320)
        self.email_entry.pack(pady=10, padx=40)
        
        # Password Input
        self.password_entry = ctk.CTkEntry(self, placeholder_text="Password", show="*", width=320)
        self.password_entry.pack(pady=10, padx=40)
        
        # Confirm Password Input
        self.confirm_password_entry = ctk.CTkEntry(self, placeholder_text="Confirm Password", show="*", width=320)
        self.confirm_password_entry.pack(pady=10, padx=40)
        
        # Register Button
        self.register_btn = ctk.CTkButton(self, text="Sign Up", width=320, fg_color="white", text_color="black", command=self.handle_registration)
        self.register_btn.pack(pady=(30, 10), padx=40)
        
        # Cancel Button
        self.cancel_btn = ctk.CTkButton(self, text="Cancel", width=320, fg_color="transparent", border_width=1, command=self.destroy)
        self.cancel_btn.pack(pady=5, padx=40)

    def handle_registration(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        confirm_password = self.confirm_password_entry.get()
        role = self.selected_role.get()
        
        # 1. Validation Checks
        if not email or not password or not confirm_password:
            messagebox.showerror("Error", "All fields are required.")
            return
            
        if password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match.")
            return
            
        if len(password) < 6:
            messagebox.showerror("Error", "Password must be at least 6 characters.")
            return
            
        # 2. Save to Database
        success = database.register_user(email, password, role)
        
        if success:
            messagebox.showinfo("Success", "Account created successfully! You can now log in.")
            self.destroy()  # Close the registration window
        else:
            messagebox.showerror("Error", "An account with this email already exists.")


class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("KSL Translator - Login")
        self.geometry("800x500")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.setup_left_panel()
        self.setup_right_panel()

    def setup_left_panel(self):
        # Left Branding Frame
        self.left_frame = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        # Logo & Text (Placeholders for your specific fonts/images)
        self.logo_label = ctk.CTkLabel(self.left_frame, text="👋", font=("Arial", 60))
        self.logo_label.pack(pady=(120, 10))
        
        self.title_label = ctk.CTkLabel(self.left_frame, text="KSL Translator", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=10)
        
        self.subtitle_label = ctk.CTkLabel(
            self.left_frame, 
            text="REAL-TIME KENYAN SIGN LANGUAGE TRANSLATION,\nPOWERED BY COMPUTER VISION", 
            font=("Arial", 10),
            text_color="gray"
        )
        self.subtitle_label.pack(pady=10)

    def setup_right_panel(self):
        # Right Authentication Frame
        self.right_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0)
        self.right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Form Elements
        self.login_label = ctk.CTkLabel(self.right_frame, text="Sign in to continue", font=("Arial", 18, "bold"))
        self.login_label.pack(pady=(60, 20), anchor="w", padx=40)
        
        # Role Toggle (Signer / Hearing User)
        self.role_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.role_frame.pack(fill="x", padx=40, pady=10)
        
        self.btn_signer = ctk.CTkButton(self.role_frame, text="Signer", width=120, fg_color="transparent", border_width=1)
        self.btn_signer.pack(side="left", padx=(0, 10))
        
        self.btn_hearing = ctk.CTkButton(self.role_frame, text="Hearing User", width=120, fg_color="transparent", border_width=1)
        self.btn_hearing.pack(side="left")
        
        # Email Input
        self.email_entry = ctk.CTkEntry(self.right_frame, placeholder_text="name@example.com", width=250)
        self.email_entry.pack(pady=(20, 10), padx=40, anchor="w")
        
        # Password Input
        self.password_entry = ctk.CTkEntry(self.right_frame, placeholder_text="........", show="*", width=250)
        self.password_entry.pack(pady=10, padx=40, anchor="w")
        
        # Log In Button
        self.login_btn = ctk.CTkButton(self.right_frame, text="Log In", width=250, fg_color="white", text_color="black", command=self.handle_login)
        self.login_btn.pack(pady=20, padx=40, anchor="w")
        
        # Bottom Links (Guest / Create Account)
        self.links_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.links_frame.pack(fill="x", padx=40, pady=10)
        
        self.guest_link = ctk.CTkLabel(self.links_frame, text="Continue as Guest", cursor="hand2", text_color="lightblue")
        self.guest_link.pack(side="left")
        
        self.register_link = ctk.CTkLabel(self.links_frame, text="Create an account", cursor="hand2", text_color="lightblue")
        self.register_link.pack(side="right")
        self.register_link.bind("<Button-1>", self.open_registration)

    def handle_login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        
        if not email or not password:
            print("Please fill in both fields.")
            return
            
        # Verify credentials using our database script
        user_role = database.verify_user(email, password)
        
        if user_role:
            print(f"Login successful! Welcome, {user_role}.")
            # TODO: Destroy login window and open the main_window.py
        else:
            print("Invalid email or password.")

    def open_registration(self, event):
        # Open the registration window and pass 'self' as the parent
        RegisterWindow(self)

if __name__ == "__main__":
    app = LoginWindow()
    app.mainloop()