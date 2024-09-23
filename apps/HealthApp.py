import time
import random

class HealthApp:
    def __init__(self, heartRate=75, sleepScore=7):
        self.heart_rate_override = False

        self._current_heart_rate = heartRate
        self._target_heart_rate = heartRate # this will update where the heart rate will simulate to, in case needed for simulation
        self._blood_pressure = (120, 80)  # systolic, diastolic
        self._sleep_score = sleepScore # below 4 concerns the system
    
    def get_heart_rate(self):
        return self._current_heart_rate
    
    def update_heart_rate(self, target):
        self.heart_rate_override = True
        self._target_heart_rate = target

    def clear_heart_rate(self):
        self.heart_rate_override = False
 
    def update_heart_rate(self, target_rate: int):
        """
        Increment or decrement the current heart rate one at a time until it reaches the target rate.
        """
        print(f"Target heart rate: {target_rate}")
        while self._current_heart_rate != target_rate:
            if self._current_heart_rate < target_rate:
                self._current_heart_rate += 1
            elif self._current_heart_rate > target_rate:
                self._current_heart_rate -= 1

            # print(f"Current heart rate: {self._current_heart_rate}")
            time.sleep(0.5)  # Adding a delay to simulate real-time updates

    def generate_random_heart_rate(self) -> int:
        """
        Generate a realistic random heart rate between 60 and 100.
        """
        return self._target_heart_rate if self.heart_rate_override else random.randint(60, 105)
    
    def start_heartbeat_monitor(self):
        """
        Continuously generates a new random heart rate and updates the current heart rate to match it,
        ensuring the target is reached before generating a new one and waiting for 2 seconds.
        """
        print('Start Heartbeat Monitor!')
        while True:
            _target_heart_rate = self.generate_random_heart_rate()
            self.update_heart_rate(_target_heart_rate)

            # Wait 2 seconds AFTER reaching the target heart rate
            print(f"Reached target heart rate: {self._current_heart_rate}. Waiting before generating new target...\n")
            time.sleep(2)  # Wait 2 seconds before generating the next target

    def getBloodPressure(self):
        return self._avgHeartbeat

    def getSleepScore(self):
        # 0 - 3 DANGER
        # 4 - 5 WARNING
        # 6 - 10 OKAY
        return self._sleep_score
    
    def start_health_app(self):
        self.start_heartbeat_monitor()

# To test the GUI
if __name__ == "__main__":
    obj = HealthApp()
    print(obj.heart_rate_override)  # Accessible
    print(obj._current_heart_rate)  # Still accessible, but by convention, should not be accessed directly
    # print(obj.__strongly_private_attribute)  # Raises AttributeError