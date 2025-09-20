#Main app with UI
import tkinter as tk
from tkinter import font

import BicepCurlTrainer
import FrontRaisesTrainer
import KneeUpsTrainer
import LateralRaises
import LegRaisesTrainer
import LungesTrainer
import PullupsTrainer
import PushupsTrainer
import SquatsTrainer
import TricepPressTrainer

def main():
    #Initialize the main window
    root = tk.Tk()
    root.title("GymGuardian - Exercise Selector")
    root.geometry("400x1000") 
    root.configure(bg="#2c3e50")

    #Define fonts
    title_font = font.Font(family="Helvetica", size=24, weight="bold")
    button_font = font.Font(family="Helvetica", size=14)
    note_font = font.Font(family="Helvetica", size=10, slant="italic")

    #Create a main frame
    main_frame = tk.Frame(root, bg="#2c3e50", padx=20, pady=20)
    main_frame.pack(expand=True, fill="both")

    #Title Label
    title_label = tk.Label(main_frame, text="GymGuardian", font=title_font, fg="#ecf0f1", bg="#2c3e50")
    title_label.pack(pady=10)

    #Subtitle Label
    subtitle_label = tk.Label(main_frame, text="Select an Exercise", font=("Helvetica", 12), fg="#bdc3c7", bg="#2c3e50")
    subtitle_label.pack(pady=(0, 20))

    #Dictionary maps button text to the 'run' function inside each imported trainer file
    exercise_functions = {
        "Bicep Curls": BicepCurlTrainer.run,
        "Front Raises": FrontRaisesTrainer.run,
        "Knee Ups": KneeUpsTrainer.run,
        "Lateral Raises": LateralRaises.run,
        "Leg Raises": LegRaisesTrainer.run,
        "Lunges": LungesTrainer.run,
        "Pull-ups": PullupsTrainer.run,
        "Push-ups": PushupsTrainer.run,
        "Squats": SquatsTrainer.run,
        "Tricep Press": TricepPressTrainer.run,
    }

    #Create buttons for each exercise
    for exercise_name, exercise_function in sorted(exercise_functions.items()):
        button = tk.Button(
            main_frame,
            text=exercise_name,
            font=button_font,
            bg="#3498db",
            fg="#ecf0f1",
            activebackground="#2980b9",
            activeforeground="#ecf0f1",
            pady=10,
            command=exercise_function 
        )
        button.pack(pady=6, fill='x')

    note_label = tk.Label(
        main_frame,
        text="Note: In the exercise window, press 'q' to quit.",
        font=note_font,
        fg="#95a5a6",
        bg="#2c3e50"
    )
    note_label.pack(pady=(20, 5))

    #Quit Button
    quit_button = tk.Button(
        main_frame,
        text="Quit Application",
        font=button_font,
        bg="#e74c3c",
        fg="#ecf0f1",
        pady=10,
        command=root.destroy
    )
    quit_button.pack(pady=5, fill='x')

    #Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()