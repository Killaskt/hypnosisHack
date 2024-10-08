import threading
import time
from apps.DriverDash import DriverDash  # Importing DriverDash from the apps folder
from apps.HealthApp import HealthApp


#
# Need to make threads to start the gui and camera
# Maybe make central data class to pass data through better, or maybe jsut some way to share data without getting too 
# is it possible to have a cool 3d circle rotational display representing head transformation on the camera viwq
# Need distracted tiem, need to account for garbage data, need to implement hypnosis, maybe make them all more robust
# since we are using the 68 predictor model maybe, we keep losing the ability to get data if full face ear to ear cant be seen
# add icon to gui to show which level we are in
# maybe add camera data to gui for demo purposes
# garbage data and ear ot ear issue is huge for camera
# hypnosis needed still
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
def action_temp_change(dashboard, temp=85, normal=70, duration=5):
    dashboard.update_temp(temp)
    dashboard.show_popup("Harsh Temp Increase!", duration)
    time.sleep(duration)
    dashboard.update_temp(normal)

def action_seat_haptics(dashboard, duration):
    dashboard.show_popup("Habtic Seat Feedback!", duration)

def action_call_driver():
    pass

def action_reducing_speed(dashboard, minspeed=50, duration=5):
    dashboard.set_speed_for_duration(minspeed, duration)
    dashboard.show_popup("Speed Decrease!", duration)
    time.sleep(duration)
    # dashboard.set_speed_for_duration(normal, 5) # it resets itself so no need
    pass

def action_steering_wheel():
    # heated steering wheel
    pass

def action_play_audio():
    pass

# ACTIONS END

def run_main_logic(dashboard, healthApp):
    """Main logic that runs concurrently with the GUI, updating heart rate"""
    run = 0
    try:
        dashboard.show_popup("Test Popup", 3)
        while True:
            # Simulate main logic running concurrently (e.g., updating heart rate)
            # Update heart rate in the dashboard (assume heart rate icon is at index 1)

            if run % 10 == 0:
                dashboard.update_icon_data(icon_index=1, data_point=healthApp.get_heart_rate())
                print(f"Main logic running... run = {run} Updated heart rate to {healthApp.get_heart_rate()}")


            if run == 20:
                print("\n\n UPDATE TEMP TEST \n\n")
                action_temp_change(dashboard=dashboard)

            if run == 40:
                print("\n\n UPDATE SPEED TEST \n\n")
                action_reducing_speed(dashboard=dashboard)
            
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
        health_app_thread.join()
        logic_thread.join()  # Wait for the logic thread to finish

        print("Program terminated cleanly.")
    # The logic thread will continue to run concurrently, while the GUI runs on the main thread