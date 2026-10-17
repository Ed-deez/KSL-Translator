from ui import translate_page
import customtkinter as ctk

# We will import the actual page classes from your other files.
# For now, we will create dummy classes below so the code runs immediately.
# import translate_page
# import history_page
# import feedback_page
# import admin_page

class MainWindow(ctk.CTk):
    def __init__(self, user_role="Signer", user_email="eddie@example.com"):
        super().__init__()
        
        self.title("KSL Translator")
        self.geometry("1000x700")
        self.configure(fg_color="#1a1a1a") # Dark background
        
        self.user_role = user_role
        self.user_email = user_email
        
        # Determine initials for the profile circle (e.g., "EW")
        # In a full app, you might fetch the actual name from the DB
        self.user_initials = "EW" 
        
        self.setup_navigation_bar()
        self.setup_main_content_area()
        
        # Dictionary to hold the different page frames
        self.pages = {}
        self.setup_pages()
        
        # Show the default page
        self.show_page("Translate")

    def setup_navigation_bar(self):
        """Creates the persistent top menu bar seen in the mockups."""
        self.nav_frame = ctk.CTkFrame(self, height=60, fg_color="#1a1a1a", corner_radius=0)
        self.nav_frame.pack(side="top", fill="x")
        self.nav_frame.pack_propagate(False) # Prevent frame from shrinking to fit content
        
        # Bottom border for the nav bar to separate it from content
        border = ctk.CTkFrame(self, height=1, fg_color="#333333", corner_radius=0)
        border.pack(side="top", fill="x")

        # Left Branding
        self.logo_label = ctk.CTkLabel(self.nav_frame, text="👋 KSL Translator", font=("Arial", 18, "bold"))
        self.logo_label.pack(side="left", padx=(20, 30))

        # Navigation Tabs
        self.tab_translate = ctk.CTkButton(self.nav_frame, text="Translate", width=80, fg_color="transparent", 
                                           text_color="white", hover_color="#333333", 
                                           command=lambda: self.show_page("Translate"))
        self.tab_translate.pack(side="left", padx=5)

        self.tab_history = ctk.CTkButton(self.nav_frame, text="History", width=80, fg_color="transparent", 
                                         text_color="gray", hover_color="#333333",
                                         command=lambda: self.show_page("History"))
        self.tab_history.pack(side="left", padx=5)

        self.tab_feedback = ctk.CTkButton(self.nav_frame, text="Feedback", width=80, fg_color="transparent", 
                                          text_color="gray", hover_color="#333333",
                                          command=lambda: self.show_page("Feedback"))
        self.tab_feedback.pack(side="left", padx=5)
        
        # Admin Tab (Only show if user has admin privileges)
        if self.user_role == "Admin":
            self.tab_admin = ctk.CTkButton(self.nav_frame, text="Admin", width=80, fg_color="transparent", 
                                              text_color="gray", hover_color="#333333",
                                              command=lambda: self.show_page("Admin"))
            self.tab_admin.pack(side="left", padx=5)

        # Right Profile Circle
        self.profile_btn = ctk.CTkButton(self.nav_frame, text=self.user_initials, width=40, height=40, 
                                         corner_radius=20, fg_color="transparent", border_width=1,
                                         border_color="gray", text_color="gray")
        self.profile_btn.pack(side="right", padx=20)

    def setup_main_content_area(self):
        """Creates the blank container where the different pages will be loaded."""
        self.content_frame = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0)
        self.content_frame.pack(side="bottom", fill="both", expand=True)

    def setup_pages(self):
        """Initializes all the pages and hides them in the content frame."""
        self.pages["Translate"] = translate_page.TranslatePage(self.content_frame)
        
        self.pages["History"] = DummyPage(self.content_frame, "History Page (Data Table goes here)")
        self.pages["Feedback"] = DummyPage(self.content_frame, "Feedback Page (Star Ratings go here)")
        self.pages["Admin"] = DummyPage(self.content_frame, "Admin Dashboard")

        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
            
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

    def show_page(self, page_name):
        """Raises the selected page to the top and updates tab text colors."""
        # Bring the requested page frame to the front
        frame = self.pages[page_name]
        frame.tkraise()
        
        # Reset all tab text colors to gray
        self.tab_translate.configure(text_color="gray")
        self.tab_history.configure(text_color="gray")
        self.tab_feedback.configure(text_color="gray")
        if hasattr(self, 'tab_admin'):
            self.tab_admin.configure(text_color="gray")
            
        # Highlight the active tab in white
        if page_name == "Translate":
            self.tab_translate.configure(text_color="white")
        elif page_name == "History":
            self.tab_history.configure(text_color="white")
        elif page_name == "Feedback":
            self.tab_feedback.configure(text_color="white")
        elif page_name == "Admin" and hasattr(self, 'tab_admin'):
            self.tab_admin.configure(text_color="white")

# --- Dummy class for testing until you build the real pages ---
class DummyPage(ctk.CTkFrame):
    def __init__(self, parent, text):
        super().__init__(parent, fg_color="transparent")
        label = ctk.CTkLabel(self, text=text, font=("Arial", 24))
        label.pack(expand=True)

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()