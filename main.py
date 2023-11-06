import pygame
import pygame_gui
import math
import matplotlib.pyplot as plt

""" Initial settings and values """

# Program settings
is_paused = False
running = True

# Screen settings
screen_width, screen_height = 1375, 700
bg_color = (0, 0, 0)
pendulum_color = (255, 255, 255)

# Pendulum properties
l1 = 15.0
l2 = 15.0
m1 = 50.0
m2 = 50.0
g = 9.81

# Define the initial state
theta1 = 0
theta2 = 0
omega1 = 0
omega2 = 0

# Constants for the simulation
dt = 0.05  # Time step
scaling_factor = 10  # Scaling factor for pendulum size

dragging_bob = None  # None when not dragging, 'bob1' or 'bob2' when dragging
initial_mouse_x, initial_mouse_y = 0, 0
initial_theta1, initial_theta2 = 0, 0

# List that stores the trail points of the second bob
trail2_points = []

# List that contains the values of the velocity of the two velocities
trail_v1_points = []
trail_v2_points = []

# Lists to store time and velocity values
time_values = []
velocity_values = []

# Lists to store time and velocity values for the graph
graph_time_values = []
graph_velocity_values = []

pygame.init()

# Pygame settings
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Double Pendulum Simulation")
clock = pygame.time.Clock()

""" Initial Calculations, using Runge-Kutta's fourth order and functions"""

# Velocities

# Velocity of the first bob
vx1 = -l1 * scaling_factor * omega1 * math.sin(theta1)
vy1 = l1 * scaling_factor * omega1 * math.cos(theta1)

# Velocity of the second bob
vx2 = -l2 * scaling_factor * omega2 * math.sin(theta2)
vy2 = l2 * scaling_factor * omega2 * math.cos(theta2)

# Total velocity
vx_total = vx1 + vx2
vy_total = vy1 + vy2

v_total = math.sqrt(vx_total**2 + vy_total**2)

# Kinetic Energy

# K.E of the first bob
ke1 = (1/2) * m1 * (vx1**2 + vy1**2)

# K.E of the second bob
ke2 = (1/2) * m2 * (vx2**2 + vy2**2)

def calculate_angle(x, y, center_x, center_y):
    return math.atan2(x - center_x, y - center_y)

# Calculates the first derivative of the angles
def derivatives(theta1, theta2, omega1, omega2):
    domega1 = (-g * (2 * m1 + m2) * math.sin(theta1) - m2 * g * math.sin(theta1 - 2 * theta2) - 2 * math.sin(theta1 - theta2) * m2 * (omega2 ** 2 * l2 + omega1 ** 2 * l1 * math.cos(theta1 - theta2))) / (l1 * (2 * m1 + m2 - m2 * math.cos(2 * theta1 - 2 * theta2)))
    domega2 = (2 * math.sin(theta1 - theta2) * (omega1 ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * math.cos(theta1) + omega2 ** 2 * l2 * m2 * math.cos(theta1 - theta2))) / (l2 * (2 * m1 + m2 - m2 * math.cos(2 * theta1 - 2 * theta2)))
    return omega1, omega2, domega1, domega2

# The Runge-Kutta's fourth order
def runge_kutta(theta1, theta2, omega1, omega2, dt):
    # First set of derivatives
    k1_theta1, k1_theta2, k1_omega1, k1_omega2 = derivatives(theta1, theta2, omega1, omega2)
    # Second set of derivatives
    k2_theta1, k2_theta2, k2_omega1, k2_omega2 = derivatives(theta1 + 0.5 * k1_theta1 * dt, theta2 + 0.5 * k1_theta2 * dt, omega1 + 0.5 * k1_omega1 * dt, omega2 + 0.5 * k1_omega2 * dt)
    # Third set of derivatives
    k3_theta1, k3_theta2, k3_omega1, k3_omega2 = derivatives(theta1 + 0.5 * k2_theta1 * dt, theta2 + 0.5 * k2_theta2 * dt, omega1 + 0.5 * k2_omega1 * dt, omega2 + 0.5 * k2_omega2 * dt)
    # Fourth set of derivatives
    k4_theta1, k4_theta2, k4_omega1, k4_omega2 = derivatives(theta1 + k3_theta1 * dt, theta2 + k3_theta2 * dt, omega1 + k3_omega1 * dt, omega2 + k3_omega2 * dt)
    
    # Updates the state variables using the weighted sum of derivatives
    theta1 += (k1_theta1 + 2 * k2_theta1 + 2 * k3_theta1 + k4_theta1) * (dt / 6)
    theta2 += (k1_theta2 + 2 * k2_theta2 + 2 * k3_theta2 + k4_theta2) * (dt / 6)
    omega1 += (k1_omega1 + 2 * k2_omega1 + 2 * k3_omega1 + k4_omega1) * (dt / 6)
    omega2 += (k1_omega2 + 2 * k2_omega2 + 2 * k3_omega2 + k4_omega2) * (dt / 6)

    return theta1, theta2, omega1, omega2

# Funtion for reseting the simulation
def reset_simulation():
    global theta1, theta2, omega1, omega2
    theta1 = 0
    theta2 = 0
    omega1 = 0
    omega2 = 0

# Function for creating a V/t type graph
def create_graph():
    # Creating the graph for the velocity
    plt.plot(graph_time_values, graph_velocity_values)

    # Adding labels and a title
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity-Time Graph')

    # Displaying the plot
    plt.show()

# Funtion for creating a plot of the movement
def create_plot():
    # X and Y coordinates
    x1_coordinates, y1_coordinates = zip(*trail_v1_points)
    x2_coordinates, y2_coordinates = zip(*trail_v2_points)

    # Plot the trail points
    plt.plot(x1_coordinates, y1_coordinates)
    plt.plot(x2_coordinates, y2_coordinates)

    # Set axis labels and a title
    plt.xlabel('X1 , X2')
    plt.ylabel('Y1, Y2')
    plt.title('Plot of the movement')

    # Display the plot
    plt.show()

def reset_values():
    global l1, l2, m1, m2, g
    l1 = 15.0
    l2 = 15.0
    m1 = 50.0
    m2 = 50.0
    g = 9.81


""" UI part """

# UI manager for the settings panel
manager = pygame_gui.UIManager((screen_width, screen_height))

# Slider for the first length
l1_slider_layout_rect = pygame.Rect((50, -250), (200, 20))
l1_slider_layout_rect.bottomleft = (50, -250)
l1_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=l1_slider_layout_rect, start_value=l1, value_range=(1, 30), manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
# Slider for the second lengyh
l2_slider_layout_rect = pygame.Rect((50, -200), (200, 20))
l2_slider_layout_rect.bottomleft = (50, -200)
l2_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=l2_slider_layout_rect, start_value=l2, value_range=(1, 30), manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
# Slider for the first mass
m1_slider_layout_rect = pygame.Rect((50, -150), (200, 20))
m1_slider_layout_rect.bottomleft = (50, -150)
m1_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=m1_slider_layout_rect, start_value=m1, value_range=(10, 200), manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
# Slider for the second mass
m2_slider_layout_rect = pygame.Rect((50, -100), (200, 20))
m2_slider_layout_rect.bottomleft = (50, -100)
m2_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=m2_slider_layout_rect, start_value=m2, value_range=(10, 200), manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
# Slider for Gravity 
g_slider_layout_rect = pygame.Rect((50, -50), (200, 20))
g_slider_layout_rect.bottomleft = (50, -50)
g_slider = pygame_gui.elements.UIHorizontalSlider(relative_rect=g_slider_layout_rect, start_value=g, value_range=(1, 20), manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})

# Labels for the sliders
l1_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, -290), (200, 20)), text=f'Length 1: {l1}m', manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
l2_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, -240), (200, 20)), text=f'Length 2: {l2}m', manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
m1_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, -190), (200, 20)), text=f'Mass 1: {m1}kg', manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
m2_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, -140), (200, 20)), text=f'Mass 2: {m2}kg', manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})
g_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((50, -90), (200, 20)), text=f'Gravity: {g}m/s^2', manager=manager, container=manager.get_root_container(), anchors={'left': 'left', 'bottom': 'bottom'})

# Labels for the changing values
v_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((screen_width - 1325, screen_height -600), (200, 20)), text=f'Total Velocity: {v_total}m/s', manager=manager, container=manager.get_root_container())
ke1_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((screen_width - 1325, screen_height -550), (200, 20)), text=f'Kinetic Energy of 1: {ke1}J', manager=manager, container=manager.get_root_container())
ke2_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((screen_width - 1325, screen_height -500), (200, 20)), text=f'Kinetic Energy of 2: {ke2}J', manager=manager, container=manager.get_root_container())

# Water Mark
pygame_gui.elements.UILabel(relative_rect=pygame.Rect((screen_width - 270, screen_height -600), (200, 20)), text=f'GitHub : StoqnToshev', manager=manager, container=manager.get_root_container())

# Reset button
reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((screen_width - 270, screen_height - 160), (200, 50)), text="Reset", manager=manager, container=manager.get_root_container())

# Pause and play button
pause_play_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((screen_width - 270, screen_height - 100), (200, 50)), text="Pause" if is_paused else "Play", manager=manager, container=manager.get_root_container())

# Graph button
graph_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((screen_width - 270, screen_height - 220), (200, 50)), text="Graph", manager=manager, container=manager.get_root_container())

# Plot button
plot_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((screen_width - 270, screen_height - 280), (200, 50)), text="Plot", manager=manager, container=manager.get_root_container())

# Reseting values button
reset_values_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((screen_width - 270, screen_height - 340), (200, 50)), text="Reset Values", manager=manager, container=manager.get_root_container())


""" Everything that happenss, while the program is running """

while running:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():

        # Quit button
        if event.type == pygame.QUIT:
            running = False

        # Changing the values in the sliders
        if event.type == pygame.USEREVENT:

            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:

                # This is for the Reset Button

                if event.ui_element == reset_button:
                    is_paused = False
                    reset_simulation()
                    trail2_points.clear()
                    trail_v1_points.clear()
                    trail_v2_points.clear()
                    time_values.clear()
                    velocity_values.clear()

                # Pause and Play button

                if event.ui_element == pause_play_button:
                    is_paused = not is_paused
                    event.ui_element.set_text("Play" if is_paused else "Pause")

                # Graph button    

                if event.ui_element == graph_button:
                    is_paused = True
                    create_graph()
                    time_values.clear()
                    velocity_values.clear()
                    graph_time_values.clear()
                    graph_velocity_values.clear()
                
                if event.ui_element == plot_button:
                    is_paused = True
                    create_plot()
                    trail_v1_points.clear()
                    trail_v2_points.clear()
                
                if event.ui_element == reset_values_button:
                    is_paused = False
                    reset_values()

            if event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == l1_slider:
                    l1 = event.value
                    l1_label.set_text(f'Length 1: {l1}m')
                elif event.ui_element == l2_slider:
                    l2 = event.value
                    l2_label.set_text(f'Length 2: {l2}m')
                elif event.ui_element == m1_slider:
                    m1 = event.value
                    m1_label.set_text(f'Mass 1: {m1}kg')
                elif event.ui_element == m2_slider:
                    m2 = event.value
                    m2_label.set_text(f'Mass 2: {m2}kg')
                elif event.ui_element == g_slider:
                    g = event.value
                    g_label.set_text(f'Gravity: {g}m/s^2')
                
        """ Dragging the pendulum """

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:

                mouse_x, mouse_y = event.pos

                # Checks if the click is on bob1 or bob2

                if abs(mouse_x - x1) <= 10 and abs(mouse_y - y1) <= 10:
                    dragging_bob = 'bob1'
                    initial_mouse_x, initial_mouse_y = mouse_x, mouse_y
                    initial_theta1, initial_theta2 = theta1, theta2

                elif abs(mouse_x - x2) <= 10 and abs(mouse_y - y2) <= 10:
                    dragging_bob = 'bob2'
                    initial_mouse_x, initial_mouse_y = mouse_x, mouse_y
                    initial_theta1, initial_theta2 = theta1, theta2

        if event.type == pygame.MOUSEMOTION:

            if dragging_bob:

                mouse_x, mouse_y = event.pos

                if dragging_bob == 'bob1':
                    # Calculate the new angle for bob1
                    delta_x = mouse_x - x1
                    delta_y = mouse_y - y1
                    new_theta1 = calculate_angle(x1 + delta_x, y1 + delta_y, screen_width / 2, screen_height / 2)
                    theta1 = new_theta1

                elif dragging_bob == 'bob2':
                    # Calculate the new angle for bob2
                    delta_x = mouse_x - x2
                    delta_y = mouse_y - y2
                    new_theta2 = calculate_angle(x2 + delta_x, y2 + delta_y, x1, y1)
                    theta2 = new_theta2

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                dragging_bob = None

        manager.process_events(event)

    screen.fill(bg_color)
    
    if not is_paused:

        # Calculates the new angles
        theta1, theta2, omega1, omega2 = runge_kutta(theta1, theta2, omega1, omega2, dt)

        # Appends the current time and velocity to the lists
        time_values.append(len(time_values) * dt)
        velocity_values.append(v_total)
        if not is_paused:  # Only update the graph lists when not paused
            graph_time_values.append(len(time_values) * dt)
            graph_velocity_values.append(v_total)

    # Calculate the positions of the pendulum bob
    x1 = l1 * scaling_factor * math.sin(theta1) + screen_width / 2
    y1 = l1 * scaling_factor * math.cos(theta1) + screen_height / 2
    x2 = x1 + l2 * scaling_factor * math.sin(theta2)
    y2 = y1 + l2 * scaling_factor * math.cos(theta2)

    # Calculate velocities
    vx1 = -l1 * scaling_factor * omega1 * math.sin(theta1)
    vy1 = l1 * scaling_factor * omega1 * math.cos(theta1)
    vx2 = -l2 * scaling_factor * omega2 * math.sin(theta2)
    vy2 = l2 * scaling_factor * omega2 * math.cos(theta2)

    # Calculate total velocity
    vx_total = vx1 + vx2
    vy_total = vy1 + vy2
    v_total = math.sqrt(vx_total**2 + vy_total**2)

    # Calculate kinetic energy
    ke1 = (1/2) * m1 * (vx1**2 + vy1**2)
    ke2 = (1/2) * m2 * (vx2**2 + vy2**2)

    if is_paused:
        v_total = 0

    # Appending the current position to the lists
    trail2_points.append((float(x2), float(y2)))
    trail_v1_points.append((float(vx1), float(vy1)))
    trail_v2_points.append((float(vx2), float(vy2)))

    # Limit the number of points to keep the trail length manageable
    max_points = 50
    if len(trail2_points) > max_points:
        trail2_points.pop(0)  # Remove the oldest point

    # Draw the pendulum on the screen
    pygame.draw.circle(screen, pendulum_color, (screen_width / 2, screen_height / 2), 10)
    pygame.draw.line(screen, pendulum_color, (screen_width // 2, screen_height // 2), (x1, y1), 5)
    pygame.draw.circle(screen, pendulum_color, (int(x1), int(y1)), 10)
    pygame.draw.line(screen, pendulum_color, (x1, y1), (x2, y2), 5)
    pygame.draw.circle(screen, pendulum_color, (int(x2), int(y2)), 10)

    for i in range(1, len(trail2_points)):
        pygame.draw.line(screen, pendulum_color, trail2_points[i - 1], trail2_points[i], 2)

    manager.update(time_delta)
    manager.draw_ui(screen)

    #Changes the values of the labels
    l1_label.update(time_delta)
    l2_label.update(time_delta)
    m1_label.update(time_delta)
    m2_label.update(time_delta)
    g_label.update(time_delta)

    v_label.set_text(f'Total Velocity: {v_total:.2f}m/s')
    ke1_label.set_text(f'Kinetic Energy of 1: {ke1:.2f} J')
    ke2_label.set_text(f'Kinetic Energy of 2: {ke2:.2f} J')

    pygame.display.flip()

pygame.quit()
