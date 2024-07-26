import tkinter as tk
from tkinter import filedialog, messagebox

class LexicalSuiteApp:
    def __init__(self, root):
        self.root = root
        root.title("TESTING")
        root.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Create the menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Create the Analysis, CSV/Excel Preview, and Settings menus
        menu_bar.add_command(label="ANALYSIS", command=self.dummy_function)
        menu_bar.add_command(label="CSV/EXCEL PREVIEW", command=self.dummy_function)
        menu_bar.add_command(label="SETTINGS", command=self.dummy_function)

        # Create the main logo and buttons
        self.logo_label = tk.Label(self.root, text="THE\nTEST\nLOOK", font=("Arial", 48), justify="center")
        self.logo_label.pack(pady=100)

        self.open_files_button = tk.Button(self.root, text="Open Files for Analysis...", command=self.open_files)
        self.open_files_button.pack(pady=10)

        self.open_folder_button = tk.Button(self.root, text="Open Folder of Files...", command=self.open_folder)
        self.open_folder_button.pack(pady=10)

    def open_files(self):
        files = filedialog.askopenfilenames(title="Open Files for Analysis")
        if files:
            messagebox.showinfo("Files Selected", f"You selected {len(files)} files.")

    def open_folder(self):
        folder = filedialog.askdirectory(title="Open Folder of Files")
        if folder:
            messagebox.showinfo("Folder Selected", f"You selected the folder: {folder}")

    def dummy_function(self):
        messagebox.showinfo("Menu Clicked", "This is a placeholder function.")

if __name__ == "__main__":
    root = tk.Tk()
    app = LexicalSuiteApp(root)
    root.mainloop()
