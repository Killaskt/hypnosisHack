import threading
import time
from queue import Queue
from apps.DriverDash import DriverDash  # Importing DriverDash from the apps folder
from apps.HealthApp import HealthApp
from apps.Camera import Camera
from apps.Serverity import Severity, SeverityActionManager, SeverityCalculator
import threading

log_lock = threading.Lock()

def log_message(message):
    with log_lock:
        print(message)

#
# Need to make threads to start the gui and camera
# Maybe make central data class to pass data through better, or maybe jsut some way to share data without getting too 
# is it possible to have a cool 3d circle rotational display representing head transformation on the camera viwq
# Need distracted tiem, need to account for garbage data, need to implement hypnosis, maybe make them all more robust
# since we are using the 68 predictor model maybe, we keep losing the ability to get data if full face ear to ear cant be seen
# add icon to gui to show which level we are in
# maybe add camera data to gui for demo purposes
#
#
# drowsy we need a confidence value from it but assume true if its 60% confident, false otherwise
# distracted true if they've been distracted in one occurence of distraction not total for more than 15s; false otherwise
# hypnosis enums or string low if they've been hypnotized for 5m, mid 10m, high 15m 
# make all these thresholds easily editable
#

        # Circular queues to store the last N yaw, pitch, and roll values
        # self.yaw_buffer = deque(maxlen=buffer_size) # (left/right) is measured as rotation around the Y-axis.
        # self.pitch_buffer = deque(maxlen=buffer_size) # (up/down) is measured as rotation around the X-axis.
        # self.roll_buffer = deque(maxlen=buffer_size) # (tilt sideways) is measured as rotation around the Z-axis.

# ACTIONS

# maybe update popup to add error, warning, normal which changes colors to red alert etc

# ACTIONS END

def run_main_logic(dashboard, healthApp, camera_queue, severityActionManager):
    """Main logic that runs concurrently with the GUI, updating heart rate"""
    run = 0
    try:
        severity = Severity()

        # Example usage for heart rate
        heart_rate = 75
        heart_rate_severity = severity.translate("heart_rate", heart_rate)
        print(f"Heart Rate Severity: {heart_rate_severity.value}")

        # Example usage for drowsiness
        drowsiness_score = 45
        drowsiness_severity = severity.translate("drowsiness", drowsiness_score)
        print(f"Drowsiness Severity: {drowsiness_severity.value}")

        # Example usage for hypnosis
        hypnosis_time = 65
        hypnosis_severity = severity.translate("hypnosis", hypnosis_time)
        print(f"Hypnosis Severity: {hypnosis_severity.value}")


        dashboard.show_popup("Test Popup", 3)
        while True:
            # Simulate main logic running concurrently (e.g., updating heart rate)
            # Update heart rate in the dashboard (assume heart rate icon is at index 1)

            if run % 10 == 0:
                dashboard.update_icon_data(icon_index=1, data_point=healthApp.get_heart_rate())
                print(f"Main logic running... run = {run} Updated heart rate to {healthApp.get_heart_rate()}")

            if run % 5 == 0:
                if not camera_queue.empty():
                    data = camera_queue.get()
                    # Simulate updating diagnostic data (you can replace these values with real data from your camera)
                    dashboard.update_diagnostic_data(hypo_fixated_time=data['fixation_start_time'], hypnotized=data['hypnotized'], distracted=data['distracted'], drowsy=data['drowsy'], drowsy_confidence=data['drowsy_confidence'], distracted_duration=data['distracted_duration'])
                    log_message(f"Distracted: {data['distracted']}, Drowsy: {data['drowsy']}, Hypnotized: {data['hypnotized']}, Yaw: {data['yaw']:.2f}, Pitch: {data['pitch']:.2f}")
                else:
                    print("none this time")

            if run == 20:
                print("\n\n UPDATE TEMP TEST \n\n")
                severityActionManager.action_temp_change()

            if run == 40:
                print("\n\n UPDATE SPEED TEST \n\n")
                severityActionManager.action_reducing_speed()
            
            run += 1
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Main logic stopped.")

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

    heart_rate = 72  # Example starting value for heart rate
    body_battery = 8
    
    # Event to signal when to stop the threads
    stop_event = threading.Event()

    # Output queues
    camera_queue = Queue()

    # Create an instance of the dashboard
    heathApp = HealthApp(heart_rate=heart_rate, body_battery=body_battery)
    camera = Camera(output_queue=camera_queue)
    dashboard = DriverDash()
    severityActionManager = SeverityActionManager(dashboard=dashboard)

    health_app_thread = threading.Thread(target=heathApp.start_health_app)
    health_app_thread.start()

    # Start the camera in its own thread
    camera_thread = threading.Thread(target=camera.run)
    camera_thread.start()

    # Start the main logic in a background thread
    logic_thread = threading.Thread(target=run_main_logic, args=(dashboard,heathApp,camera_queue,severityActionManager))
    logic_thread.start()

    try:
        # Run the GUI on the main thread (pygame GUI should run on the main thread)
        dashboard.show_gui()

    except KeyboardInterrupt:
        print("GUI interrupted")

    finally:
        # Stop the background thread gracefully
        stop_event.set()  # Signal the stop event to terminate the background thread
        health_app_thread.join()
        camera_thread.join()
        logic_thread.join()  # Wait for the logic thread to finish

        print("Program terminated cleanly.")
    # The logic thread will continue to run concurrently, while the GUI runs on the main thread


    # biggest things right now are to remove imshow and just have camera run in bg, and maybe add icons and stuff to gui