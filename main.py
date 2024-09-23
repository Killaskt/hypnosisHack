import threading
import time
from apps.DriverDash import DriverDash  # Importing DriverDash from the apps folder
from apps.HealthApp import HealthApp


#
# Need to make threads to start the gui and camera
# Maybe make central data class to pass data through better, or maybe jsut some way to share data without getting too 
#

def run_main_logic(dashboard, healthApp):
    """Main logic that runs concurrently with the GUI, updating heart rate"""
    run = 0
    try:
        while True:
            # Simulate main logic running concurrently (e.g., updating heart rate)
            # Update heart rate in the dashboard (assume heart rate icon is at index 1)

            if run % 10 == 0:
                dashboard.update_icon_data(icon_index=1, data_point=healthApp.get_heart_rate())
                print(f"Main logic running... run = {run} Updated heart rate to {healthApp.get_heart_rate()}")
            
            run += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Main logic stopped.")

if __name__ == "__main__":
    print("Ultimte Hypnosis Detector starting...")

    # BEFORE
    # simulate heart rate
    # camera drowsiness
    # camera 


    # AFTER (ACTIONS)
    # simulate speed
    # simulate temp
    # simulate volume (audio)

    heart_rate = 72  # Example starting value for heart rate
    sleep_score = 8
    
    # Event to signal when to stop the threads
    stop_event = threading.Event()

    # Create an instance of the dashboard
    heathApp = HealthApp(heartRate=heart_rate, sleepScore=sleep_score)
    dashboard = DriverDash()


    health_app_thread = threading.Thread(target=heathApp.start_health_app)
    health_app_thread.start()

    # Start the main logic in a background thread
    logic_thread = threading.Thread(target=run_main_logic, args=(dashboard,heathApp,))
    logic_thread.start()

    try:
        # Run the GUI on the main thread (pygame GUI should run on the main thread)
        dashboard.show_gui()

    except KeyboardInterrupt:
        print("GUI interrupted")

    finally:
        # Stop the background thread gracefully
        stop_event.set()  # Signal the stop event to terminate the background thread
        logic_thread.join()  # Wait for the logic thread to finish

        print("Program terminated cleanly.")
    # The logic thread will continue to run concurrently, while the GUI runs on the main thread