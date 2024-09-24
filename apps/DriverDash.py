import pygame
import os
import math
import time
import random

class DriverDash:
    def __init__(self, HealthApp=None):
        # Initialize Pygame
        pygame.init()

        # Project Necessary
        self.HealthData = HealthApp

        # Screen dimensions
        self.screen_width = 1024
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Car Driver\'s Screen')

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 100, 255)
        self.GRAY = (50, 50, 50)
        self.YELLOW = (255, 255, 0)
        self.LIGHT_GRAY = (180, 180, 180)
        self.GREEN = (0, 255, 0)
        self.POPUP_BG = (50, 50, 50)  # Slightly darker background for popup

        # Font
        self.font = pygame.font.SysFont('Arial', 40, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 25)
        self.tiny_font = pygame.font.SysFont('Arial', 15)

        # Variables for car data
        # New attributes in __init__
        self.speed = 60  # Start the speed at a default value near 60
        self.min_speed = 65  # Editable min speed
        self.max_speed = 75  # Editable max speed
        self.speed_change_rate = 0.5  # Rate at which speed changes per update
        self.update_interval = 0.1  # Delay between updates in seconds
        self.last_update_time = pygame.time.get_ticks() / 1000  # Track the time since the last update
        self.override_speed = None  # To handle speed overrides
        self.override_end_time = None  # To handle the duration of the override

        self.rpm = 1.0  # Start RPM at 1.0
        self.max_rpm = 8.0  # Start RPM at 1.0
        self.rpm_target = 3.0  # Initial target RPM
        self.rpm_change_rate = 0.05  # How much RPM changes per update
        self.rpm_update_interval = 0.1  # Time delay between RPM updates (in seconds)
        self.last_rpm_update_time = pygame.time.get_ticks() / 1000  # To track the time since last RPM update


        self.temperature = 90  # Example temperature
        self.fuel_level = 0.8  # Fuel level as percentage (0 to 1)

        # Popup settings
        self.popup_msg = ""
        self.popup_start_time = 0
        self.popup_duration = 5  # seconds

        # Icons and lights
        self.icons = []  # Will hold all lights/icons to display

        # Diagnostic data from the camera (default values)
        self.hypo_fixated_time = 0.0
        self.hypnotized = 'none'
        self.distracted = False
        self.drowsy = False
        self.drowsy_confidence = 0.0
        self.distracted_duration = 0.0

        # Load engine icon image with error handling and recoloring
        self.engine_icon = self.verify_image_path('icons/engine-icon.png')  # Use the correct path to your image
        self.heart_rate_icon = self.verify_image_path('icons/heartbeat-icon.png')  # Use the correct path to your image

    def verify_image_path(self, image_path):
        temp_icon = None
        
        if os.path.exists(image_path):
            temp_icon = pygame.image.load(image_path).convert_alpha()
            temp_icon = pygame.transform.scale(temp_icon, (40, 40))  # Resize it as needed
        else:
            print(f"Error: Image file '{image_path}' not found.")
            temp_icon = None  # Prevent crash if image isn't found
        
        return temp_icon

    def recolor_image(self, image, new_color):
        """Recolors an image by modifying its pixels to match the new_color"""
        image = image.copy()  # Copy the original image
        color_key = image.map_rgb((255, 255, 255))  # Assuming the base color is white

        # Lock the image for pixel manipulation
        pixels = pygame.PixelArray(image)
        pixels.replace(color_key, new_color)  # Replace white with the new color
        del pixels  # Unlock the image

        return image

    def draw_speed_gauge(self, center_x, center_y, value, max_value=120, label="MPH", color=None):
        """Draw a speed gauge with ticks representing 0, 10, 20, ..., 120."""
        
        # Draw the arc for the gauge
        pygame.draw.arc(self.screen, self.GRAY, (center_x - 120, center_y - 120, 240, 240), 0, math.pi, 10)
        pygame.draw.arc(self.screen, self.BLACK, (center_x - 110, center_y - 110, 220, 220), 0, math.pi, 5)

        # Calculate the angle for the needle
        angle = math.pi * (value / max_value)  # Value mapped to 0-180 degrees
        needle_length = 100
        needle_x = center_x - needle_length * math.cos(angle)  # Flip horizontally for gauge layout
        needle_y = center_y - needle_length * math.sin(angle)
        pygame.draw.line(self.screen, color, (center_x, center_y), (needle_x, needle_y), 5)

        # Draw ticks along the arc, mapped to 0, 10, 20, ..., 120
        num_ticks = int(max_value / 10) + 1  # Number of tick marks for increments of 10

        for i in range(num_ticks):
            tick_value = i * 10  # Tick value (0, 10, 20, ...)
            tick_angle = math.pi * (tick_value / max_value)  # Angle for each tick

            # Calculate inner and outer coordinates for each tick
            tick_length = 15 if i % 2 == 0 else 7  # Longer tick every other one
            outer_x = center_x - 110 * math.cos(tick_angle)  # Flip horizontally
            outer_y = center_y - 110 * math.sin(tick_angle)
            inner_x = center_x - (110 - tick_length) * math.cos(tick_angle)
            inner_y = center_y - (110 - tick_length) * math.sin(tick_angle)

            pygame.draw.line(self.screen, self.WHITE, (outer_x, outer_y), (inner_x, inner_y), 2)

            # Label every 10 units (0, 10, 20, ...)
            tick_label = self.tiny_font.render(str(tick_value), True, self.WHITE)
            self.screen.blit(tick_label, (outer_x - tick_label.get_width() // 2, outer_y - tick_label.get_height() // 2))

        # Digital readout in the center
        text = self.font.render(f'{int(value)}', True, self.WHITE)
        self.screen.blit(text, (center_x - text.get_width() // 2, center_y - text.get_height() // 2))

        # Label for the gauge (e.g., MPH)
        label_text = self.small_font.render(label, True, self.WHITE)
        self.screen.blit(label_text, (center_x - label_text.get_width() // 2, center_y + 80))

    def draw_rpm_gauge(self, center_x, center_y, value, max_value=8, label="RPM", color=None):
        """Draw an RPM gauge with ticks representing 0, 1, 2, ..., 8."""
        
        # Draw the arc for the gauge
        pygame.draw.arc(self.screen, self.GRAY, (center_x - 120, center_y - 120, 240, 240), 0, math.pi, 10)
        pygame.draw.arc(self.screen, self.BLACK, (center_x - 110, center_y - 110, 220, 220), 0, math.pi, 5)

        # Calculate the angle for the needle
        angle = math.pi * (value / max_value)  # Value mapped to 0-180 degrees
        needle_length = 100
        needle_x = center_x - needle_length * math.cos(angle)  # Flip horizontally for gauge layout
        needle_y = center_y - needle_length * math.sin(angle)
        pygame.draw.line(self.screen, color, (center_x, center_y), (needle_x, needle_y), 5)

        # Draw ticks along the arc, mapped to 0, 1, 2, ..., 8
        num_ticks = int(max_value) + 1  # Number of tick marks for increments of 1

        for i in range(num_ticks):
            tick_value = i  # Tick value (0, 1, 2, ..., 8)
            tick_angle = math.pi * (tick_value / max_value)  # Angle for each tick

            # Calculate inner and outer coordinates for each tick
            tick_length = 15 if i % 2 == 0 else 7  # Longer tick every other one
            outer_x = center_x - 110 * math.cos(tick_angle)  # Flip horizontally
            outer_y = center_y - 110 * math.sin(tick_angle)
            inner_x = center_x - (110 - tick_length) * math.cos(tick_angle)
            inner_y = center_y - (110 - tick_length) * math.sin(tick_angle)

            pygame.draw.line(self.screen, self.WHITE, (outer_x, outer_y), (inner_x, inner_y), 2)

            # Label every unit (0, 1, 2, ...)
            tick_label = self.tiny_font.render(str(tick_value), True, self.WHITE)
            self.screen.blit(tick_label, (outer_x - tick_label.get_width() // 2, outer_y - tick_label.get_height() // 2))

        # Digital readout in the center
        text = self.font.render(f'{int(value)}', True, self.WHITE)
        self.screen.blit(text, (center_x - text.get_width() // 2, center_y - text.get_height() // 2))

        # Label for the gauge (e.g., RPM)
        label_text = self.small_font.render(label, True, self.WHITE)
        self.screen.blit(label_text, (center_x - label_text.get_width() // 2, center_y + 80))

    def update_rpm(self):
        """Update RPM smoothly within the range of 1 to 3, with a realistic fluctuation"""
        current_time = pygame.time.get_ticks() / 1000  # Get the current time

        # Only update the RPM after the RPM update interval
        if current_time - self.last_rpm_update_time >= self.rpm_update_interval:

            # Gradually approach the target RPM
            if self.rpm < self.rpm_target:
                self.rpm = min(self.rpm + self.rpm_change_rate, self.rpm_target)
            elif self.rpm > self.rpm_target:
                self.rpm = max(self.rpm - self.rpm_change_rate, self.rpm_target)

            # If RPM reaches the target, pick a new target between 1 and 3
            if abs(self.rpm - self.rpm_target) < 0.01:  # Close enough to target
                self.rpm_target = random.uniform(1.0, 3.0)  # New target between 1 and 3

            # Update the last RPM update time
            self.last_rpm_update_time = current_time

    def draw_fuel_temperature_bars(self):
        """Draw fuel and temperature bars with ticks at the bottom of the screen"""
        # Fuel Bar
        fuel_label = self.small_font.render('Fuel', True, self.WHITE)
        pygame.draw.rect(self.screen, self.WHITE, (100, 500, 300, 20), 2)  # Shifted left
        pygame.draw.rect(self.screen, self.GREEN, (100, 500, 300 * self.fuel_level, 20))
        self.screen.blit(fuel_label, (100, 530))

        # Draw ticks for fuel
        for tick in range(0, 11):  # Fuel percentage ticks from 0 to 100%
            tick_x = 100 + (30 * tick)
            pygame.draw.line(self.screen, self.WHITE, (tick_x, 500), (tick_x, 520), 2)
            if tick % 2 == 0:
                fuel_tick_label = self.tiny_font.render(f'{tick * 10}%', True, self.WHITE)
                self.screen.blit(fuel_tick_label, (tick_x - 10, 525))

        # Temperature Bar
        temp_label = self.small_font.render('Temp', True, self.WHITE)
        pygame.draw.rect(self.screen, self.WHITE, (624, 500, 300, 20), 2)  # Shifted right
        pygame.draw.rect(self.screen, (self.RED if self.temperature > 85 else (self.GREEN if self.temperature > 50 else self.BLUE)), (624, 500, 300 * self.temperature / 120, 20))
        self.screen.blit(temp_label, (624, 530))

        # Draw ticks for temperature
        for tick in range(0, 13, 2):  # Temperature ticks from 0 to 120 degrees C
            tick_x = 624 + (25 * tick)
            pygame.draw.line(self.screen, self.WHITE, (tick_x, 500), (tick_x, 520), 2)
            temp_tick_label = self.tiny_font.render(f'{tick * 10}°', True, self.WHITE)
            self.screen.blit(temp_tick_label, (tick_x - 10, 525))

    def show_popup(self, message, duration):
        """Display a temporary popup message"""
        self.popup_msg = message
        self.popup_duration = duration
        self.popup_start_time = pygame.time.get_ticks() / 1000  # Set the time when the popup is triggered

    def draw_popup(self):
        """Draw the popup message if active, with centered text and larger popup height"""
        if self.popup_msg:
            current_time = pygame.time.get_ticks() / 1000  # Get the current time in seconds
            # print(f"\n\n Popup active - {current_time}s, start_time: {self.popup_start_time}s \n\n")
            
            # Check if the popup duration has expired
            if (current_time - self.popup_start_time) < self.popup_duration:
                # Render the popup text
                popup_text = self.font.render(self.popup_msg, True, self.WHITE)
                
                # Determine the size of the popup based on text width and desired height
                popup_width = popup_text.get_width() + 40  # Add padding around the text
                popup_height = popup_text.get_height() + 30  # Larger popup height
                popup_x = (self.screen_width - popup_width) // 2  # Center the popup horizontally
                popup_y = (self.screen_height - popup_height) // 2  # Center the popup vertically

                # Create the popup rectangle
                popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
                
                # Draw the popup background
                pygame.draw.rect(self.screen, self.POPUP_BG, popup_rect)
                
                # Draw the popup border
                pygame.draw.rect(self.screen, self.LIGHT_GRAY, popup_rect, 2)

                # Center the text within the popup
                text_x = popup_x + (popup_width - popup_text.get_width()) // 2
                text_y = popup_y + (popup_height - popup_text.get_height()) // 2
                self.screen.blit(popup_text, (text_x, text_y))
            else:
                # print(f"\n\n Popup duration exceeded. Hiding popup.\n\n")
                # Reset the popup message once the time is up
                self.popup_msg = ""

    def draw_main_readouts(self):
        """Draw the main MPH and Temp readout at the top center"""
        mph_text = self.font.render(f'{int(self.speed)} MPH', True, self.WHITE)
        self.screen.blit(mph_text, (self.screen_width // 2 - mph_text.get_width() // 2, 50))

        # Draw small icons or indicators for warnings
        direction_text = self.small_font.render(f'SW', True, self.WHITE)
        self.screen.blit(direction_text, (self.screen_width // 2 - direction_text.get_width() // 2, 100))

        range_text = self.small_font.render(f'Range: 455 mi', True, self.WHITE)
        self.screen.blit(range_text, (self.screen_width // 2 - range_text.get_width() // 2, 130))

        temp_text = self.small_font.render(f'{int(self.temperature)}°F', True, self.WHITE)
        self.screen.blit(temp_text, (self.screen_width // 2 - temp_text.get_width() // 2, 160))

    def draw_icons(self):
        """Draw the lights/icons (like engine) from the icon list and display real-time data"""
        for i, icon in enumerate(self.icons):
            x_pos = self.screen_width // 2 - 60 + (i * 80)  # Spacing between icons
            y_pos = 460
            if icon['type'] == 'image':
                self.screen.blit(icon['image'], (x_pos, y_pos))  # Use image instead of circle
            else:
                pygame.draw.circle(self.screen, icon['color'], (x_pos, y_pos + 20), 30)
                label_text = self.tiny_font.render(icon['label'], True, self.BLACK)
                self.screen.blit(label_text, (x_pos - label_text.get_width() // 2, y_pos + 10))
            
            # If the icon has a data point, display it next to the icon
            if 'data_point' in icon:
                data_text = self.small_font.render(f'{icon["data_point"]}', True, self.WHITE)
                if data_text != None:
                    self.screen.blit(data_text, (x_pos + 50, y_pos + 10))  # Display data next to the icon

    def add_icon(self, label=None, color=None, image=None, data_point=None):
        """Add new icons/lights to the display (like engine light). Supports both images and colors, and optional data points"""
        if image:
            if data_point:
                self.icons.append({"type": "image", "image": image, "data_point": data_point})
            else:
                self.icons.append({"type": "image", "image": image})
        else:
            self.icons.append({"type": "color", "label": label, "color": color, "data_point": data_point})

    def update_icon_data(self, icon_index, data_point):
        """Update the data point for an existing icon"""
        if 0 <= icon_index < len(self.icons):
            self.icons[icon_index]['data_point'] = data_point

    def update_diagnostic_data(self, hypo_fixated_time=None, hypnotized=None, distracted=None, drowsy=None, drowsy_confidence=None, distracted_duration=None):
        """Update the diagnostic data values"""
        if hypo_fixated_time is not None:
            self.hypo_fixated_time = hypo_fixated_time
        if hypnotized is not None:
            self.hypnotized = hypnotized
        if distracted is not None:
            self.distracted = distracted
        if drowsy is not None:
            self.drowsy = drowsy
        if drowsy_confidence is not None:
            self.drowsy_confidence = drowsy_confidence
        if distracted_duration is not None:
            self.distracted_duration = distracted_duration

    def draw_diagnostic_data(self):
        """Draw the diagnostic data on the top-right of the screen"""
        diagnostic_text = [
            f'Hypo Fixated Time: {self.hypo_fixated_time:.2f}s',
            f'Hypnotized: {self.hypnotized}',
            f'Distracted: {"Yes" if self.distracted else "No"}',
            f'Drowsy: {"Yes" if self.drowsy else "No"}',
            f'Drowsy Confidence: {self.drowsy_confidence:.2f}s',
            f'distracted duration: {self.distracted_duration:.2f}s'
        ]
        
        # Starting position for the text (top-right corner)
        start_x = self.screen_width - 300
        start_y = 50

        for i, line in enumerate(diagnostic_text):
            diagnostic_surface = self.tiny_font.render(line, True, self.WHITE)
            self.screen.blit(diagnostic_surface, (start_x, start_y + i * 20))


    def show_gui(self):
        """Main function to run the GUI"""
        running = True
        clock = pygame.time.Clock()

        # Add initial icons
        if self.engine_icon:
            self.add_icon(image=self.engine_icon, color=self.GREEN)

        # real-time data point icon
        if self.heart_rate_icon:
            self.add_icon(image=self.heart_rate_icon, color=self.RED, data_point=72)  # Initial heart rate

        while running:
            self.screen.fill(self.BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Check for keypress events
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # If 'q' is pressed
                        running = False


            self.update_rpm()

            # Update and draw elements
            self.draw_speed_gauge(200, 300, self.speed, self.max_speed, 'MPH', self.BLUE)  # Shifted left
            self.draw_rpm_gauge(824, 300, self.rpm, self.max_rpm, 'RPM', self.BLUE)  # Shifted right
            self.draw_main_readouts()
            self.draw_fuel_temperature_bars()
            self.draw_icons()
            self.draw_popup()

            # Draw diagnostic data
            self.draw_diagnostic_data()

            # Simulate heart rate update (real-time data)
            # updated_heart_rate = 72 + int((pygame.time.get_ticks() / 1000) % 30)  # Just for demo purposes
            # self.update_icon_data(1, updated_heart_rate)  # Update the heart rate icon data

            # Simulate speed change (for testing)
            # Inside your main loop (e.g., in `show_gui`):
            current_time = pygame.time.get_ticks() / 1000

            # Only update the speed after the update interval
            if current_time - self.last_update_time >= self.update_interval:
                
                # Check if an override is active
                if self.override_speed is not None and current_time < self.override_end_time:
                    # Gradually approach the override speed
                    if self.speed < self.override_speed:
                        self.speed = min(self.speed + self.speed_change_rate, self.override_speed)
                    elif self.speed > self.override_speed:
                        self.speed = max(self.speed - self.speed_change_rate, self.override_speed)
                else:
                    # If override has expired or not active, reset it
                    if self.override_end_time is not None and current_time >= self.override_end_time:
                        self.override_speed = None  # Reset override once time is up
                    
                    # Gradually approach a random target speed within the range of 65 to 75
                    target_speed = random.uniform(65, 75)
                    if self.speed < target_speed:
                        self.speed = min(self.speed + self.speed_change_rate, target_speed)
                    elif self.speed > target_speed:
                        self.speed = max(self.speed - self.speed_change_rate, target_speed)
                
                # Update the last update time
                self.last_update_time = current_time



            # self.speed = (self.speed + 0.5) % self.max_speed
            self.rpm = (self.rpm + 50) % self.max_rpm

            # Update the display
            pygame.display.flip()

            # Control the frame rate
            clock.tick(30)

        pygame.quit()

    def update_temp(self, temp):
        self.temperature = temp

    def set_speed_for_duration(self, speed, duration):
        """Set a specific speed for a certain duration (in seconds)"""
        self.override_speed = speed
        self.override_end_time = pygame.time.get_ticks() / 1000 + duration  # Set end time for the override


# To test the GUI
if __name__ == "__main__":
    dashboard = DriverDash()
    dashboard.show_gui()
