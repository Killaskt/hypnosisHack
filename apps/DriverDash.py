import pygame
import os
import math

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
        self.speed = 0
        self.rpm = 1000
        self.max_speed = 160
        self.max_rpm = 8000  # Updated RPM limit to 8000
        self.temperature = 90  # Example temperature
        self.fuel_level = 0.8  # Fuel level as percentage (0 to 1)

        # Popup settings
        self.popup_msg = ""
        self.popup_start_time = 0
        self.popup_duration = 5  # seconds

        # Icons and lights
        self.icons = []  # Will hold all lights/icons to display

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

    def draw_semi_gauge(self, center_x, center_y, value, max_value, label, color):
        """Draw a semi-circular gauge with a needle from 0 to 180 degrees, flipped horizontally"""
        pygame.draw.arc(self.screen, self.GRAY, (center_x - 120, center_y - 120, 240, 240), 0, math.pi, 10)
        pygame.draw.arc(self.screen, self.BLACK, (center_x - 110, center_y - 110, 220, 220), 0, math.pi, 5)

        # Draw needle
        angle = math.pi * (value / max_value)  # 0 to 180 degrees scale
        needle_length = 100
        needle_x = center_x - needle_length * math.cos(angle)  # Flipped horizontally
        needle_y = center_y - needle_length * math.sin(angle)
        pygame.draw.line(self.screen, color, (center_x, center_y), (needle_x, needle_y), 5)

        # Draw ticks along the arc
        for tick in range(0, 181, 20):  # Every 20 degrees
            tick_angle = math.radians(tick)
            tick_length = 15 if tick % 40 == 0 else 7  # Longer ticks at multiples of 40
            outer_x = center_x - 110 * math.cos(tick_angle)  # Flipped
            outer_y = center_y - 110 * math.sin(tick_angle)
            inner_x = center_x - (110 - tick_length) * math.cos(tick_angle)
            inner_y = center_y - (110 - tick_length) * math.sin(tick_angle)
            pygame.draw.line(self.screen, self.WHITE, (outer_x, outer_y), (inner_x, inner_y), 2)

        # Digital readout in the center
        text = self.font.render(f'{int(value)}', True, self.WHITE)
        self.screen.blit(text, (center_x - text.get_width() // 2, center_y - text.get_height() // 2))

        # Gauge label
        label_text = self.small_font.render(label, True, self.WHITE)
        self.screen.blit(label_text, (center_x - label_text.get_width() // 2, center_y + 80))

    def draw_fuel_temperature_bars(self):
        """Draw fuel and temperature bars with ticks at the bottom of the screen"""
        # Fuel Bar
        fuel_label = self.small_font.render('Fuel', True, self.WHITE)
        pygame.draw.rect(self.screen, self.WHITE, (100, 500, 300, 20), 2)  # Shifted left
        pygame.draw.rect(self.screen, self.YELLOW, (100, 500, 300 * self.fuel_level, 20))
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
        pygame.draw.rect(self.screen, self.RED, (624, 500, 300 * self.temperature / 120, 20))
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
        self.popup_start_time = time.time()
        self.popup_duration = duration

    def draw_popup(self):
        """Draw the popup message if active, with subtle background and smaller size"""
        if self.popup_msg and time.time() - self.popup_start_time < self.popup_duration:
            popup_text = self.font.render(self.popup_msg, True, self.WHITE)
            popup_rect = pygame.Rect(300, 200, 400, 40)
            pygame.draw.rect(self.screen, self.POPUP_BG, popup_rect)  # Subtle background
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, popup_rect, 2)  # Border
            self.screen.blit(popup_text, (310, 205))

    def draw_main_readouts(self):
        """Draw the main MPH and Temp readout at the top center"""
        mph_text = self.font.render(f'{int(self.speed)} MPH', True, self.WHITE)
        self.screen.blit(mph_text, (self.screen_width // 2 - mph_text.get_width() // 2, 50))

        # Draw small icons or indicators for warnings
        direction_text = self.small_font.render(f'SW', True, self.WHITE)
        self.screen.blit(direction_text, (self.screen_width // 2 - direction_text.get_width() // 2, 100))

        range_text = self.small_font.render(f'Range: 455 mi', True, self.WHITE)
        self.screen.blit(range_text, (self.screen_width // 2 - range_text.get_width() // 2, 130))

        temp_text = self.small_font.render(f'{int(self.temperature)}°C', True, self.WHITE)
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

    def show_gui(self):
        """Main function to run the GUI"""
        running = True
        clock = pygame.time.Clock()

        # Add initial icons
        if self.engine_icon:
            self.add_icon(image=self.engine_icon, color=self.GREEN)

        # real-time data point icon
        if self.heart_rate_icon:
            self.add_icon(image=self.heart_rate_icon, label="yolo", color=self.RED, data_point=72)  # Initial heart rate

        while running:
            self.screen.fill(self.BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update and draw elements
            self.draw_semi_gauge(200, 300, self.speed, self.max_speed, 'MPH', self.BLUE)  # Shifted left
            self.draw_semi_gauge(824, 300, self.rpm, self.max_rpm, 'RPM', self.BLUE)  # Shifted right
            self.draw_main_readouts()
            self.draw_fuel_temperature_bars()
            self.draw_icons()
            self.draw_popup()

            # Simulate heart rate update (real-time data)
            updated_heart_rate = 72 + int((pygame.time.get_ticks() / 1000) % 30)  # Just for demo purposes
            self.update_icon_data(1, updated_heart_rate)  # Update the heart rate icon data

            # Simulate speed change (for testing)
            self.speed = (self.speed + 0.5) % self.max_speed
            self.rpm = (self.rpm + 50) % self.max_rpm

            # Update the display
            pygame.display.flip()

            # Control the frame rate
            clock.tick(30)

        pygame.quit()

# To test the GUI
if __name__ == "__main__":
    dashboard = DriverDash()
    dashboard.show_gui()
