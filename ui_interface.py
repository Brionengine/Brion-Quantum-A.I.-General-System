# ui/interface.py

import tkinter as tk
from tkinter import messagebox

def launch_ui():
    """
    Launches the user interface.
    """
    def on_submit():
        user_input = entry.get()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a command.")
        else:
            # Process the user input
            messagebox.showinfo("Success", f"Command received: {user_input}")

    root = tk.Tk()
    root.title("Quantum Fusion Interface")

    label = tk.Label(root, text="Enter your command:")
    label.pack(pady=10)

    entry = tk.Entry(root, width=50)
    entry.pack(pady=5)

    submit_button = tk.Button(root, text="Submit", command=on_submit)
    submit_button.pack(pady=10)

    root.mainloop()
