class MyClass:
    def __init__(self):
        self._current_heart_rate = 42
        self._target_heart_rate = 42 # this will update where the heart rate will simulate to, in case needed for simulation
        self._blood_pressure = (120, 80)  # systolic, diastolic
        self._sleep_score = 10 # below 4 concerns the system

    def u(self):
        return "This is a public method"
    
    def get_heart_beat(self):
        return self._avg_heartbeat
 
    def getBloodPressure(self):
        return self._avgHeartbeat

    def getSleepScore(self):
        return self._sleep_score
    
    def update_heart_rate(self, target_rate: int):
        """
        Increment or decrement the current heart rate one at a time until it reaches the target rate.
        """
        print(f"Target heart rate: {target_rate}")
        while self.current_heart_rate != target_rate:
            if self.current_heart_rate < target_rate:
                self.current_heart_rate += 1
            elif self.current_heart_rate > target_rate:
                self.current_heart_rate -= 1

            print(f"Current heart rate: {self.current_heart_rate}")
            time.sleep(0.1)  # Adding a delay to simulate real-time updates

    def start_heartbeat_monitor(self):
        """
        Continuously generates a new random heart rate and updates the current heart rate to match it,
        ensuring the target is reached before generating a new one and waiting for 2 seconds.
        """
        while True:
            target_heart_rate = self.generate_random_heart_rate()
            self.update_heart_rate(target_heart_rate)

            # Wait 2 seconds AFTER reaching the target heart rate
            print(f"Reached target heart rate: {self.current_heart_rate}. Waiting before generating new target...\n")
            time.sleep(2)  # Wait 2 seconds before generating the next target


    

    def _private_method(self):
        return "This is a weakly private method"

    def __strongly_private_method(self):
        return "This is a strongly private method"

# Testing
obj = MyClass()
print(obj.public_attribute)  # Accessible
print(obj._private_attribute)  # Still accessible, but by convention, should not be accessed directly
# print(obj.__strongly_private_attribute)  # Raises AttributeError

# Accessing strongly private attribute via mangling:
print(obj._MyClass__strongly_private_attribute)  # Accessible via name mangling
