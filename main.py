import threading
import time
from apps.DriverDash import DriverDash  # Importing DriverDash from the apps folder
from apps.HealthApp import HealthApp


#
# Need to make threads to start the gui and camera
# Maybe make central data class to pass data through better, or maybe jsut some way to share data without getting too 
#
if __name__ == "__main__":
    print("Ultimate Hypnosis Detector starting...")

    # BEFORE
    # simulate heart rate
    # camera drowsiness
    # camera 


    # AFTER (ACTIONS)
    # simulate speed
    # simulate temp
    # simulate volume (audio)

    # Create an instance of the dashboard
    dashboard = DriverDash()
    heathApp = HealthApp()

    # Create and start a thread to run the GUI
    gui_thread = threading.Thread(target=dashboard.show_gui)
    gui_thread.start()

    # Main program logic here
    heart_rate = 72  # Example starting value for heart rate

    try:
        while True:
            # Simulate main logic running concurrently (e.g., updating heart rate)


            # Update heart rate in the dashboard (assume heart rate icon is at index 1)
            dashboard.update_icon_data(icon_index=1, data_point=heart_rate)

            print(f"Main logic running... Updated heart rate to {heart_rate}")

    except KeyboardInterrupt:
        print("Main program stopped.")

