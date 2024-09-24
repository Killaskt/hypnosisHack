from enum import Enum
import time

class SeverityLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "med"
    HIGH = "high"


class Severity:
    def __init__(self):
        # You can customize thresholds per metric if needed
        self.severity_thresholds = {
            "heart_rate": {"none": 20, "low": 40, "medium": 60},
            "body_battery": {"none": 20, "low": 40, "medium": 60},
            "drowsiness": {"none": 20, "low": 40, "medium": 60},
            "distraction": {"none": 20, "low": 40, "medium": 60},
            "hypnosis": {"none": 20, "low": 40, "medium": 60},
        }

    def _get_severity_level(self, value, metric):
        thresholds = self.severity_thresholds.get(metric, {"none": 20, "low": 40, "medium": 60})

        if value <= thresholds["none"]:
            return SeverityLevel.NONE
        elif thresholds["none"] < value <= thresholds["low"]:
            return SeverityLevel.LOW
        elif thresholds["low"] < value <= thresholds["medium"]:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.HIGH

    def translate(self, metric, value):
        """
        Translate a raw value into severity level for a given metric.
        :param metric: The metric type (e.g., 'heart_rate', 'drowsiness')
        :param value: The raw value to be translated
        :return: The corresponding severity level as a SeverityLevel enum
        """
        return self._get_severity_level(value, metric)


class SeverityActionManager:
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.action_map = {
            SeverityLevel.NONE: self.action_none,
            SeverityLevel.LOW: self.action_low,
            SeverityLevel.MEDIUM: self.action_medium,
            SeverityLevel.HIGH: self.action_high
        }

    def execute_action(self, severity_level):
        """
        Executes the action corresponding to the severity level.
        """
        action = self.action_map.get(severity_level, self.default_action)
        action()

    def action_none(self):
        print("Action: No action required.")

    def action_low(self):
        print("Action: Low-level action executed.")
        self.action_temp_change(temp=85, normal=70, duration=5)

    def action_medium(self):
        print("Action: Medium-level action executed.")
        self.action_seat_haptics(duration=3)

    def action_high(self):
        print("Action: High-level action executed.")
        self.action_call_driver(duration=5)

    def default_action(self):
        print("Action: Default action executed (unrecognized severity).")

    def action_temp_change(self, temp=85, normal=70, duration=5):
        self.dashboard.update_temp(temp)
        self.dashboard.show_popup("Harsh Temp Increase!", duration)
        time.sleep(duration)
        self.dashboard.update_temp(normal)

    def action_seat_haptics(self, duration=5):
        self.dashboard.show_popup("Habtic Seat Feedback!", duration)

    def action_call_driver(self, duration=5):
        self.dashboard.show_popup("Calling the Driver!", duration)

    def action_reducing_speed(self, minspeed=50, duration=5):
        self.dashboard.set_speed_for_duration(minspeed, duration)
        self.dashboard.show_popup("Speed Decrease!", duration)
        time.sleep(duration)

    def action_steering_wheel(self, duration=5):
        self.dashboard.show_popup("Heating the steering wheel!", duration)

    def action_play_audio(self, duration=5):
        self.dashboard.show_popup("Playing the audio!", duration)


class SeverityCalculator:
    def __init__(self, severity_instance, action_manager):
        """
        Initialize the calculator with a Severity instance and an Action Manager instance.
        """
        self.severity = severity_instance
        self.action_manager = action_manager

    def calculate_severity(self, metrics):
        """
        Takes in a dictionary of metrics and values, calculates the overall severity level, 
        and triggers the corresponding action.

        :param metrics: A dictionary of metric names (e.g., 'heart_rate') and their corresponding values.
        :return: None
        """
        overall_severity = SeverityLevel.NONE  # Start with a default severity level

        for metric, value in metrics.items():
            metric_severity = self.severity.translate(metric, value)
            
            # Compare to find the highest severity level
            if metric_severity.value > overall_severity.value:
                overall_severity = metric_severity

        # Pass the calculated severity to the action manager
        self.action_manager.execute_action(overall_severity)

# # Example Usage
# dashboard = Dashboard()
# severity = Severity()
# action_manager = SeverityActionManager(dashboard)
# calculator = SeverityCalculator(severity, action_manager)

# # Example input values (metrics)
# metrics = {
#     "heart_rate": 55,
#     "body_battery": 35,
#     "drowsiness": 70
# }

# # Calculate the severity and execute the corresponding action
# calculator.calculate_severity(metrics)
