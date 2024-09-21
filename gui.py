import tkinter as tk
from tkinter import ttk

# Editable variables for easy customization
speed_value = 65
rpm_value = 3
speed_unit = "Km/h"
fuel_label = "Fuel"
door_status_locked = "Door Locked"
door_status_unlocked = "Door Unlocked"
button_bg = "#4c4f5a"  # Darker background for buttons
button_fg = "#000"  # Lighter text for buttons for contrast
window_bg = "#1e1f29"  # Dark background for the window
needle_speed_color = "#ffcc00"  # Yellow needle for speedometer
needle_rpm_color = "#33cc33"  # Green needle for tachometer
needle_width = 4
circle_outline_color = "#ffffff"  # White for circle outlines

# Function to toggle door lock status
def toggle_door_status():
    if door_btn.config('text')[-1] == door_status_unlocked:
        door_btn.config(text=door_status_locked)
    else:
        door_btn.config(text=door_status_unlocked)

# Main window setup
root = tk.Tk()
root.title("Car Dashboard")
root.geometry("900x500")  # Initial size of the window
root.configure(bg=window_bg)

# Ensure the layout resizes properly when the window is resized
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)

# Frame for the dashboard
dashboard_frame = tk.Frame(root, bg=window_bg)
dashboard_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=20, pady=20)

# Speedometer Canvas (Left Side)
speed_canvas = tk.Canvas(dashboard_frame, width=300, height=300, bg=window_bg, highlightthickness=0)
speed_canvas.create_oval(50, 50, 250, 250, outline=circle_outline_color, width=3)
speed_canvas.create_text(150, 140, text=str(speed_value), font=("Helvetica", 36), fill="white")
speed_canvas.create_text(150, 200, text=speed_unit, font=("Helvetica", 14), fill="white")
speed_canvas.create_line(150, 150, 200, 100, fill=needle_speed_color, width=needle_width)
speed_canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Tachometer Canvas (Right Side)
rpm_canvas = tk.Canvas(dashboard_frame, width=300, height=300, bg=window_bg, highlightthickness=0)
rpm_canvas.create_oval(50, 50, 250, 250, outline=circle_outline_color, width=3)
rpm_canvas.create_text(150, 140, text=str(rpm_value), font=("Helvetica", 36), fill="white")
rpm_canvas.create_text(150, 200, text=fuel_label, font=("Helvetica", 14), fill="white")
rpm_canvas.create_line(150, 150, 200, 120, fill=needle_rpm_color, width=needle_width)
rpm_canvas.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

# Car status frame in the center
car_frame = tk.Frame(dashboard_frame, bg=window_bg)
car_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Car status label
car_label = tk.Label(car_frame, text="Car Status", bg=window_bg, fg="white", font=("Helvetica", 16))
car_label.pack(pady=10)

# Door lock/unlock button
door_btn = tk.Button(car_frame, text=door_status_unlocked, font=("Helvetica", 12), bg=button_bg, fg=button_fg, command=toggle_door_status)
door_btn.pack(pady=5)

# Lower section for control buttons
control_frame = tk.Frame(root, bg=window_bg)
control_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky="nsew")

# List of control button names
control_buttons = ["DASHBOARD", "AC", "MUSIC", "MAP"]

# Create control buttons dynamically
for text in control_buttons:
    btn = tk.Button(control_frame, text=text, font=("Helvetica", 12), bg=button_bg, fg=button_fg, width=12, height=2, bd=0, relief=tk.FLAT)
    btn.pack(side=tk.LEFT, padx=20)

# Date and Time label at the bottom
date_time_label = tk.Label(control_frame, text="Date - Time", bg=window_bg, fg="white", font=("Helvetica", 12))
date_time_label.pack(side=tk.BOTTOM, pady=10)

# Run the application
root.mainloop()
