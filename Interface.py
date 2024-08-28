import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from Algorithm import Algorithm  # Import your Algorithm class

class EmotionApp:
    def __init__(self, root):
        self.root = root
        root.title("Emotion Analysis")
        root.geometry("800x600")

        self.create_widgets()

        # Initialize the Algorithm class
        self.algorithm = Algorithm()
        self.fig = None  # Initialize fig to store the graph figure

    def create_widgets(self):
        # Create the menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Create the Analysis, CSV/EXCEL Preview, and Settings menus
        menu_bar.add_command(label="ANALYSIS", command=self.dummy_function)
        menu_bar.add_command(label="CSV/EXCEL PREVIEW", command=self.dummy_function)
        menu_bar.add_command(label="SETTINGS", command=self.dummy_function)

        # Create the main logo and buttons
        self.logo_label = tk.Label(self.root, text="Emotion\nAnalysis\nApp", font=("Arial", 48), justify="center")
        self.logo_label.pack(pady=(50, 20))  # Adjust padding for better layout

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        self.open_files_button = tk.Button(button_frame, text="Open Files for Analysis...", command=self.open_files)
        self.open_files_button.grid(row=0, column=0, padx=5, pady=5)

        self.open_folder_button = tk.Button(button_frame, text="Open Folder of Files...", command=self.open_folder)
        self.open_folder_button.grid(row=0, column=1, padx=5, pady=5)

        # Create the progress bars
        self.chunk_progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.chunk_progress.pack(pady=(10, 5))

        self.overall_progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.overall_progress.pack(pady=(5, 10))

        # Add a button to download the graph
        self.download_button = tk.Button(self.root, text="Download Graph", command=self.download_graph)
        self.download_button.pack(pady=10)  # Ensure this button is visible below the progress bar

    def update_chunk_progress(self, current, total):
        self.chunk_progress["maximum"] = total
        self.chunk_progress["value"] = current
        self.root.update_idletasks()

    def update_overall_progress(self, current, total):
        self.overall_progress["maximum"] = total
        self.overall_progress["value"] = current
        self.root.update_idletasks()

    def open_files(self):
        files = filedialog.askopenfilenames(title="Open Files for Analysis")
        if files:
            for file in files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        text = f.read()
                        print(f"Length of text: {len(text)}")  # Debugging: print the length of the text
                except Exception as e:
                    messagebox.showerror("File Error", f"An error occurred while opening the file: {e}")
                    continue

                # Generate the emotion graph with progress updates
                self.fig = self.algorithm.plot_emotion_graph(
                    text,
                    progress_callback_chunk=self.update_chunk_progress,
                    progress_callback_overall=self.update_overall_progress
                )

                # Notify the user that the graph is ready to be downloaded
                messagebox.showinfo("Graph Ready", "The graph has been generated and is ready for download.")

    def open_folder(self):
        folder = filedialog.askdirectory(title="Open Folder of Files")
        if folder:
            messagebox.showinfo("Folder Selected", f"You selected the folder: {folder}")

    def dummy_function(self):
        messagebox.showinfo("Menu Clicked", "This is a placeholder function.")

    def download_graph(self):
        if self.fig:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if file_path:
                try:
                    self.fig.savefig(file_path)
                    messagebox.showinfo("Success", f"Graph saved successfully at {file_path}")
                except Exception as e:
                    messagebox.showerror("Save Error", f"An error occurred while saving the file: {e}")
        else:
            messagebox.showwarning("No Graph", "No graph to save. Please generate a graph first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
