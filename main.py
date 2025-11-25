from typing import List, Any
import pygame
import numpy as np
import random


DEBUGGING = False


'''
READ ME:
This is version 2 of the simulator
it uses the base tick approach where the simulation has a minimum time step
and with calculate the number of physics steps required per frame based on DELTA TIME (dt)

Fixed issues:
planets circles are now to scale
changing dt no longer implements distortions to the simulation
code is cleaner and easier to understand
'''

# Colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set up pygame screen
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
CENTRE_X = SCREEN_WIDTH // 2
CENTRE_Y = SCREEN_HEIGHT // 2
FPS = 30

# Clock
clock = pygame.time.Clock()

# CONSTANTS
G = 6.67e-11
dt = 450
BASETICK = 225
pixel_scale = 0.5e-6

# Initialise Pygame
pygame.init()

# Font
font = pygame.font.Font(None, 30)


# Classes

# Body
class Body:
    def __init__(self, name: str, mass: float, orbital_radius: float, velocity: float, object_radius : float):
        # Attributes
        self.name = name
        self.mass = mass
        # Spawning an object enforces centre Y and radius is implied by x-axis separation
        self.position = np.array([orbital_radius, 0], dtype=np.float64)
        # Object velocity at spawn is forced vertical on y-axis for simplicity
        self.velocity = np.array([0, velocity], dtype=np.float64)
        # Acceleration
        self.acceleration = np.array([0, 0], dtype=np.float64)
        # The planets use an array to store past positions
        self.past_positions = []
        # Every Object will have a size dependent on its mass
        self.size = object_radius


    # Methods

    def draw_body(self):
        global move_offset_x, move_offset_y
        global CENTRE_X, CENTRE_Y
        # Converts real values to a scale model
        draw_x = int(self.position[0] * pixel_scale) + CENTRE_X + move_offset_x
        draw_y = int(self.position[1] * pixel_scale) + CENTRE_Y + move_offset_y
        radius = max(2, int(self.size * pixel_scale))

        pygame.draw.circle(screen, WHITE, (draw_x, draw_y), radius)

        # Draw the objects name next to the object
        text_surface = font.render(self.name, True, WHITE)
        text_x = draw_x + 3
        text_y = draw_y - (self.size * pixel_scale* 2) - 3
        # Draw the name
        screen.blit(text_surface, (text_x, text_y))

    def draw_trajectories(self, draw_trajectories: bool):
        global move_offset_x, move_offset_y

        if len(self.past_positions) > 0:
            if draw_trajectories:
                step = 255 / len(self.past_positions)
                col = 0

                # Draw from the past positions
                for i in range(len(self.past_positions) - 1):
                    x1 = int(self.past_positions[i][0] * pixel_scale) + CENTRE_X
                    y1 = int(self.past_positions[i][1] * pixel_scale) + CENTRE_Y
                    x2 = int(self.past_positions[i + 1][0] * pixel_scale) + CENTRE_X
                    y2 = int(self.past_positions[i + 1][1] * pixel_scale) + CENTRE_Y

                    #pygame.draw.line(screen, WHITE, (x1 + move_offset_x, y1 + move_offset_y), (x2 + move_offset_x, y2 + move_offset_y), 1)

                    pygame.draw.line(screen, (col,col,col),
                                     (x1 + move_offset_x, y1 + move_offset_y),
                                     (x2 + move_offset_x, y2 + move_offset_y),
                                     1)

                    col = col + step

    def check_for_mouse_click(self, mouse_pos: tuple):
        global move_offset_x, move_offset_y
        # Check if the mouse coordinates are within the radius
        check_radius = max(2, int(self.size * pixel_scale))
        check_x = int(self.position[0] * pixel_scale) + CENTRE_X + move_offset_x
        check_y = int(self.position[1] * pixel_scale) + CENTRE_Y + move_offset_y
        if (check_x - check_radius) <= mouse_pos[0] <= (check_x + check_radius)and (check_y - check_radius) <= mouse_pos[1] <= (check_y + check_radius):
            # return the clicked on body
            return self
        else:
            return None


# Moon
class Sat(Body):
    def __init__(self, parent_body: Body, name: str, mass: float, relative_orbital_radius: float, relative_velocity: float, object_radius : float):
        # Initialise super class
        super().__init__(name=name, mass=mass, orbital_radius=relative_orbital_radius, velocity=0, object_radius=object_radius)
        self.parent_body = parent_body
        # Convert all measurements from local (relative to parent) to global (relative to system)
        self.position = np.array([relative_orbital_radius, 0], dtype=np.float64) + self.parent_body.position + np.array([self.parent_body.size, 0], dtype=np.float64)
        self.velocity = np.array([0, relative_velocity], dtype=np.float64) + self.parent_body.velocity



# Global Functions
'''
A-Level Approach but slow, numpy approach is much faster
def calculate_gravitational_acceleration(body_list: List[Body]) -> None:
    # This function does the following:
    # Iterate through all the bodies, calculate the force, apply is to a vector
    # For every body in the simulation

    # Rest all acceleration, it doesn't stack
    for current_body in body_list:
        # Reset the numpy array
        current_body.acceleration[:] = 0

    for body1 in body_list:
        for body2 in body_list:
            # Only apply physics if the bodies are different
            if body1 != body2:

                # Get separation and vector
                direction = body2.position - body1.position
                distance = max(np.linalg.norm(direction), 1.0e-20)
                # Get the force vector with r^ = r / |r| UNIT vector equation
                unit_vector = direction / distance

                # calculate the force
                force_magnitude = (G * body1.mass * body2.mass) / distance**2
                # Take the force and make it into an acceleration vector
                force_vector = force_magnitude * unit_vector
                # Apply forces to body F = m * a -> a = F/m
                body1.acceleration += (force_vector / body1.mass)
'''

# Optimised acceleration computation function using numpy maths only, but it does the same thing as the original function
def calculate_gravitational_acceleration(body_list: List[Body]) -> None:
    # Reset accelerations for all bodies
    for b in body_list:
        b.acceleration[:] = 0.0

    N = len(body_list)
    if N < 2:
        return

    # Build vectorized arrays (N, dims) for positions and (N,) for masses
    positions = np.array([b.position for b in body_list], dtype=float)
    masses = np.array([b.mass for b in body_list], dtype=float)

    # Pairwise separation vectors r_ij = r_j - r_i --> shape (N, N, dims)
    r = positions[None, :, :] - positions[:, None, :]

    # Compute squared distances |r_ij|^2 --> shape (N, N)
    dist_sq = np.sum(r * r, axis=2)

    # Avoid division by zero (self-interactions only)
    dist_sq = np.maximum(dist_sq, 1e-20)

    # Compute 1 / r^3 = (1 / r^2)^(3/2)
    inv_r3 = dist_sq ** -1.5

    # Explicitly zero out diagonal so bodies don't accelerate themselves
    np.fill_diagonal(inv_r3, 0.0)

    # Precompute G * m_j for all j
    factors = G * masses  # shape (N,)

    # Compute acceleration contributions:
    # acc[i,j] = G * m_j * (r_j - r_i) / |r_ij|^3
    acc = (r * inv_r3[..., None]) * factors[None, :, None]

    # Sum over j to get total acceleration for each body i
    acc = np.sum(acc, axis=1)

    # Write the acceleration vectors back into the Body objects
    for i, b in enumerate(body_list):
        b.acceleration[:] = acc[i]

def gravitational_potential_wells(bodies_list: List[Body], toggle_potentials: bool) -> None:
    if not toggle_potentials:
        return

    for b in bodies_list:

        # Screen coordinates with camera
        draw_x = int(b.position[0] * pixel_scale) + CENTRE_X + move_offset_x
        draw_y = int(b.position[1] * pixel_scale) + CENTRE_Y + move_offset_y

        # Potential at surface of the body
        phi_surface = -G * b.mass / b.size

        # Fractions of surface potential
        fractions = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.1, 0.05]
        color_step = (255 / (len(fractions)))
        min_col = 255 // 4
        col = 255

        for f in fractions:

            φ = phi_surface * f
            r_world = -G * b.mass / φ
            r_pixels = int(r_world * pixel_scale)

            if 2 < r_pixels:
                pygame.draw.circle(screen, (col, 0, 0), (draw_x, draw_y), r_pixels, 2)

            # Colour must be greater than a certain threshold
            if col >= min_col:
                col -= color_step

# Update physics
def update_simulation(current_simulation_dt: int, body_list: List[Body]) -> None:

    """
    The algorithm uses the base tick approach, the program has a minimum physics step
    and dt is the delta time per fram, the engine will calculate physics steps
    in units of base ticks
    Takes a dt variable for the engines current delta time, and calculates the physics
    per minimum time step to ensure changing delta time doesn't implement distortions
    """

    for i in range(current_simulation_dt // BASETICK):
        calculate_gravitational_acceleration(body_list)
        for current_body in body_list:
            # Now calculate the new positions
            current_body.velocity += current_body.acceleration * BASETICK
            current_body.position += current_body.velocity * BASETICK

            # Add the point to the past positions but make sure it's never more than 5000
            current_body.past_positions.append(current_body.position.copy())
            if len(current_body.past_positions) >= 100:
                current_body.past_positions.pop(0)

# Lock a planet and follow it
def lock_planet(check_body, locked):
    global CENTRE_X, CENTRE_Y
    global move_offset_x, move_offset_y
    global paused

    if locked and check_body is not None:
        # Predict where the body will be THIS FRAME
        if not paused:
            predicted_pos = check_body.position + check_body.velocity * dt
        else:
            predicted_pos = check_body.position

        CENTRE_X = int(round(-(predicted_pos[0] * pixel_scale) + SCREEN_WIDTH / 2 + move_offset_x))
        CENTRE_Y = int(round(-(predicted_pos[1] * pixel_scale) + SCREEN_HEIGHT / 2 + move_offset_y))

# Text Renderer
def render_text(message: str, location: tuple[int, int]) -> None:
    """
    This function renders any text on the main screen at a given coordinate

    :param message: Text to display
    :param location: Screen Coordinates to render
    :return: None
    """
    # Create the text to blit
    text_surface = font.render(message, True, WHITE)
    screen.blit(text_surface, location)

# System Clock
def SystemClock(seconds: int, current_simulation_dt) -> None:
    # Get the global time
    days = seconds // 86400
    seconds %= 86400

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    clock_str = f'{days}days {hours}hours'
    render_text(clock_str, (int(3/4 * SCREEN_WIDTH) , int(1/8 * SCREEN_HEIGHT)))
    render_text(f'{current_simulation_dt} seconds per frame',(int(3/4 * SCREEN_WIDTH) , int(1/8 * SCREEN_HEIGHT) + 20))

# Pause/Play Simulation
def draw_pause_button(toggle_pause: bool):
    # Draw a pause button and if you click there it toggles pause
    if toggle_pause:
        render_text("Simulation Paused", (int(1/32 * SCREEN_WIDTH), 50))
    else:
        render_text("Simulation Running...", (int(1/32 * SCREEN_WIDTH), 50))

    render_text("Press [SPACE] to toggle pause", (int(1/32 * SCREEN_WIDTH), 70))

# Print object Data
def output_body_data(selected_body: Body) -> None:
    # For safety if referenced before assignment
    rel_velocity = 0
    if selected_body is not None:
        mass = selected_body.mass
        velocity = int(np.linalg.norm(selected_body.velocity))
        if type(selected_body) == Sat:
            rel_velocity = int(np.linalg.norm(selected_body.velocity - selected_body.parent_body.velocity))

        render_text(f'Data for body: {selected_body.name}', (int(1/32 * SCREEN_WIDTH), int(2/3 * SCREEN_HEIGHT)))
        render_text(f'Body Mass: {mass}kg', (int(1 / 32 * SCREEN_WIDTH), int(2 / 3 * SCREEN_HEIGHT) + 20))
        render_text(f'Body physical Radius: {selected_body.size}m',(int(1 / 32 * SCREEN_WIDTH), int(2 / 3 * SCREEN_HEIGHT) + 40))
        if type(selected_body) == Sat:
            render_text(f'Global Velocity(Relative to system): {velocity}ms^-1',(int(1 / 32 * SCREEN_WIDTH), int(2 / 3 * SCREEN_HEIGHT) + 80))
            render_text(f'Local Velocity(Relative to Parent Body): {rel_velocity}ms^-1',(int(1 / 32 * SCREEN_WIDTH), int(2 / 3 * SCREEN_HEIGHT) + 100))
            render_text(f'Orbiting around: {selected_body.parent_body.name}', (int(1 / 32 * SCREEN_WIDTH), int(2 / 3 * SCREEN_HEIGHT) + 60))
        else:
            render_text(f'Body Velocity: {velocity}ms^-1',
                        (int(1 / 32 * SCREEN_WIDTH), int(2 / 3 * SCREEN_HEIGHT) + 60))

def help_screen(show_help: bool) -> None:
    if show_help:
        render_text("KeyBinds:", (int(1/5 * SCREEN_WIDTH) + 100, int(1/8 * SCREEN_HEIGHT)))
        render_text("BACKSLASH : Show/Hide Planet Trajectories", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 20))
        render_text("Space : Pause/Play Simulation", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 40))
        render_text("[ and ] : Increase/ Decrease Time per frame", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 60))
        render_text("- and = : Zoom In/Out", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 80))
        render_text("TAB : Reset Simulation View", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 100))
        render_text("Arrow Keys : Move Around Simulation", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 120))
        render_text("Click on a Body to show its data AND Shift + Left click to focus on a body", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 140))
        render_text("Press ; to show/hide input Mode, if a parent body is selected...", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 160))
        render_text("...Click a red button next to the field you wish to enter...", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 180))
        render_text("...The details you enter will be relative to THAT body then press ENTER to create the body", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 200))
        render_text("PLEASE NOTE, when adding a body, if it's a satellite, Orbital radius is from the parent body's surface", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 220))
        render_text("Click Escape once to go back to the system select menu, and Escape again to quit", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 240))
        render_text("COMMA to show/hide Gravitaional Potential Wells, the closer the lines the stronger the gravity", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 260))
        render_text("P : Toggle Presenter Mode, every 10s it switches to a random body", (int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 280))
        render_text("PERIOD : Show/Hide this Help Screen at any time",(int(1 / 5 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 360))

def presentation_mode() -> Body | Sat | Any:
    """
    When this function is called, once every 10 seconds, it will select a random integer
    get the corresponding body at that index in the list bodies and returns it

    :returns Body | Sat
    """
    random_index = int(random.randint(0, len(bodies) - 1))
    body_to_return = bodies[random_index]
    return body_to_return

def control_presenter_radius(current_body: Body) -> None:
    """
    While in presenter mode, this function is called once per frame
    It Takes in a body object that's being focussed on
    If it's a body, it ensures its draw size isn't bigger than 1/3 of the screen and not smaller than 1/16 of the screen
    If it's a satellite, it ensures that the parent body it's orbiting will be visable by hanging pixel scale according to
    the distance between sat and parent
    """
    global pixel_scale
    global move_offset_x, move_offset_y, CENTRE_X, CENTRE_Y

    if clicked_body is not None:
        if type(current_body) == Sat:
            draw_distance = np.linalg.norm(current_body.parent_body.position - current_body.position) * pixel_scale
            parent_body_draw_size = current_body.parent_body.size * pixel_scale
            if draw_distance < (SCREEN_WIDTH * 1/8):
                pixel_scale *= 2
            elif draw_distance > (SCREEN_WIDTH * 1/2):
                pixel_scale /= 2
        else:
            draw_size = current_body.size * pixel_scale
            if draw_size >= (SCREEN_WIDTH * 1/3):
                pixel_scale /= 2
            elif draw_size <= (SCREEN_WIDTH * 1/16):
                pixel_scale *= 2

def add_body_mode(is_adding_body, selected_body: Body) -> None:
    global add_mass_string, add_name_string, add_radius_string, add_object_radius_string, add_velocity_string
    if is_adding_body:
        if selected_body is None:
            parent =  None
        else:
            parent = selected_body.name
        render_text(f'Parent Body: {parent}', (int(SCREEN_WIDTH * 2/4) + 100, int(SCREEN_HEIGHT * 3/4)))
        render_text(f'Orbital Radius (in meters): {add_radius_string} ', (int(SCREEN_WIDTH * 2/4) + 100, int(SCREEN_HEIGHT * 3/4) + 20))
        render_text(f'Orbital Velocity (in ms^-1): {add_velocity_string}', (int(SCREEN_WIDTH * 2/4) + 100, int(SCREEN_HEIGHT * 3/4) + 40))
        render_text(f'Objects Physical size (radius in meters): {add_object_radius_string}', (int(SCREEN_WIDTH * 2/4) + 100, int(SCREEN_HEIGHT * 3/4) + 60))
        render_text(f'Object Name: {add_name_string}', (int(SCREEN_WIDTH * 2/4) + 100, int(SCREEN_HEIGHT * 3/4) + 80))
        render_text(f'Object Mass (in kg): {add_mass_string}', (int(SCREEN_WIDTH * 2/4) + 100, int(SCREEN_HEIGHT * 3/4) + 100))

        # Draw boxes next to these text areas
        for i in range(5):
            pygame.draw.rect(screen, RED, (int(SCREEN_WIDTH * 2/4) + 50, int(SCREEN_HEIGHT * 3/4) + 20 + (i * 20), 23, 23), 2)

def which_input_box(current_mouse_pos: tuple) ->None | int:
    # Determine which input box the user is typing on
    # If the x bound is correct
    if int(SCREEN_WIDTH * 2/4) + 73 >= current_mouse_pos[0] >= int(SCREEN_WIDTH * 2/4) + 50:
        # Check which box determined by the y location
        if int(SCREEN_HEIGHT * 3 / 4) + 40 >= current_mouse_pos[1] >= int(SCREEN_HEIGHT * 3 / 4) + 20:
            # Box 1
            return 1
        if int(SCREEN_HEIGHT * 3 / 4) + 60 >= current_mouse_pos[1] >= int(SCREEN_HEIGHT * 3 / 4) + 40:
            # Box 1
            return 2
        if int(SCREEN_HEIGHT * 3 / 4) + 80 >= current_mouse_pos[1] >= int(SCREEN_HEIGHT * 3 / 4) + 60:
            # Box 1
            return 3
        if int(SCREEN_HEIGHT * 3 / 4) + 100 >= current_mouse_pos[1] >= int(SCREEN_HEIGHT * 3 / 4) + 80:
            # Box 1
            return 4
        if int(SCREEN_HEIGHT * 3 / 4) + 120 >= current_mouse_pos[1] >= int(SCREEN_HEIGHT * 3 / 4) + 100:
            return 5
    return None

def instantiate_body(current_clicked_body: Body) -> None:
    global add_mass_string, add_name_string, add_radius_string, add_object_radius_string, add_velocity_string
    body_name = add_name_string
    body_mass = add_mass_string
    body_radius = add_radius_string
    body_object_radius = add_object_radius_string
    body_velocity = add_velocity_string
    try:
        # if it's a sat make the sat relative to parent(clicked_body)
        if clicked_body is not None:
            bodies.append(Sat(current_clicked_body, str(body_name), float(body_mass), float(body_radius), float(body_velocity), float(body_object_radius)))
        else:
            bodies.append(Body(body_name, float(body_mass), float(body_radius), float(body_velocity), float(body_object_radius)))
    except:
        print("Error, fields were incorrectly entered, couldn't instantiate body")

    add_mass_string = ""
    add_name_string = ""
    add_radius_string = ""
    add_object_radius_string = ""
    add_velocity_string = ""

def select_system_menu():
    render_text("Please Select a system:", (CENTRE_X - 100, CENTRE_Y - 80))
    # Draw boxes next to these text areas
    for i in range(3):
        pygame.draw.rect(screen, RED,(CENTRE_X - 100, CENTRE_Y + (i * 20), 23, 23), 2)
    render_text("The Solar System", (CENTRE_X - 70, CENTRE_Y))
    render_text("Earth Moon System", (CENTRE_X - 70, CENTRE_Y + 23))
    render_text("Empty System: SandBox (Build Your Own)", (CENTRE_X - 70, CENTRE_Y + 46))
    render_text("Press Escape to Quit from this menu", (CENTRE_X - 100, CENTRE_Y + 100))

def which_system_box(current_mouse_pos: tuple) -> None | int:
    # Checks the mouse pos and finds which box it correlates to
    if CENTRE_X - 100 < current_mouse_pos[0] < CENTRE_X - 77:
        if CENTRE_Y < current_mouse_pos[1] < CENTRE_Y + 23:
            return 1
        if CENTRE_Y + 23 < current_mouse_pos[1] < CENTRE_Y + 46:
            return 2
        if CENTRE_Y + 46 < current_mouse_pos[1] < CENTRE_Y + 69:
            return 3
    return None


# Instances

# Stars
Sun = Body("Sun", 1.9885e30, 0, 0, 6.957e8)


# PLANETS (orbital radius & velocity around Sun)
Mercury = Body("Mercury", 3.3011e23, 5.79e10, 4.79e4, 2.4397e6)
Venus = Body("Venus", 4.867e24, 1.082e11, 3.5e4, 6.0518e6)
Earth = Body("Earth", 5.97237e24, 1.496e11, 2.978e4, 6.371e6)
Earth_rel_centre = Body("Earth", 5.97237e24, 0, 0, 6.371e6)
Mars = Body("Mars", 6.4171e23, 2.279e11, 2.41e4, 3.3895e6)
Jupiter = Body("Jupiter", 1.898e27, 7.7857e11, 1.307e4, 6.9911e7)
Saturn = Body("Saturn", 5.683e26, 1.4335e12, 9.68e3, 5.8232e7)
Uranus = Body("Uranus", 8.681e25, 2.8725e12, 6.8e3, 2.5362e7)
Neptune = Body("Neptune", 1.024e26, 4.495e12, 5.43e3, 2.4622e7)

# MOONS (orbital radius & velocity relative to parent body center)
Moon = Sat(Earth, "The Moon", 7.3477e22, 3.844e8, 1.022e3, 1.7374e6)
Moon_rel_centre = Sat(Earth_rel_centre, "The Moon", 7.3477e22, 3.844e8, 1.022e3, 1.7374e6)
Phobos = Sat(Mars, "Phobos", 1.0659e16, 9.378e6, 2.14e3, 11266)
Deimos = Sat(Mars, "Deimos", 1.4762e15, 2.3459e7, 1.35e3, 6200)

Io = Sat(Jupiter, "Io", 8.9319e22, 4.217e8, 1.73e4, 1.8216e6)
Europa = Sat(Jupiter, "Europa", 4.799e22, 6.709e8, 1.37e4, 1.5608e6)
Ganymede = Sat(Jupiter, "Ganymede", 1.4819e23, 1.070e9, 1.08e4, 2.6341e6)
Callisto = Sat(Jupiter, "Callisto", 1.075e23, 1.8827e9, 8.2e3, 2.4103e6)

Titan = Sat(Saturn, "Titan", 1.3452e23, 1.2219e9, 5.6e3, 2.5755e6)
Enceladus = Sat(Saturn, "Enceladus", 1.08e20, 2.379e8, 1.26e4, 252100)

Titania = Sat(Uranus, "Titania", 3.527e21, 4.36e8, 3.7e3, 788900)
Oberon = Sat(Uranus, "Oberon", 3.014e21, 5.83e8, 3.15e3, 761400)

Triton = Sat(Neptune, "Triton", 2.14e22, 3.547e8, 4.39e3, 1.3534e6)

# DWARF PLANETS (orbit Sun)
Pluto = Body("Pluto", 1.303e22, 5.906e12, 4.74e3, 1.1883e6)
Charon = Sat(Pluto, "Charon", 1.586e21, 1.9571e7, 210.8, 606000)

Ceres = Body("Ceres", 9.383e20, 4.14e11, 1.79e4, 469700)
Eris = Body("Eris", 1.66e22, 1.016e13, 3.0e3, 1.163e6)
Haumea = Body("Haumea", 4.006e21, 6.45e12, 4.9e3, 816000)
Makemake = Body("Makemake", 3.1e21, 6.85e12, 4.4e3, 715000)

# Asteroids
Vesta = Body("4 Vesta", 2.59076e20, 3.53e11, 19340.0, 525.4e3)
Ida = Body("243 Ida", 4.2e16, 4.456e11, 16.92e3, 29000)
Psyche = Body("16 Psyche", 2.293e19, 4.97e11, 15.65e3, 1.20e5)

# COMETS (orbit Sun)
Halley = Body("Halley's Comet", 2.2e14, 5.3e12, 689, 5500)
HaleBopp = Body("Comet Hale-Bopp", 2.2e14, 5.29e13, 111.5, 30000)
C67P = Body("Comet 67P", 1e13, 5.87e11, 1.5e4, 2000)

# ARTIFICIAL SATELLITES (radius from parent surface)
CAPSTONE = Sat(Moon, "CAPSTONE", 25, 1.5e6, 1.0e3, 2)
CAPSTONE_rel_centre = Sat(Moon_rel_centre, "CAPSTONE", 25, 1.5e6, 1.0e3, 2)
ARTEMISP1 = Sat(Moon, "ARTEMIS P1", 500, 1.8e6, 1.5e3, 2)
ARTEMISP1_rel_centre = Sat(Moon_rel_centre, "ARTEMIS P1", 500, 1.8e6, 1.5e3, 2)
Cassini = Sat(Saturn, "Cassini", 2125.0, 11646e3, 23297.6, 6.7)
Juno = Sat(Jupiter, "Juno", 3625.0, 4200e3, 41342.8, 3.5)
Titan_LowOrbit_Sat = Sat(Titan, "Titan_LowOrbiter", 5.0e2, 100e3, 1831.8, 2.0)
Dactyl = Sat(Ida, "Dactyl", 1.3e12, 90e3, 4.85, 800)
Dawn_Vesta = Sat(Vesta, "Dawn (at Vesta)", 1.2e3, 500e3, 151.0, 2.5)

Hubble = Sat(Earth, "Hubble Space Telescope", 11110, 6.9e6, 7.6e3, 5)
Hubble_rel_centre = Sat(Earth_rel_centre, "Hubble", 1.1e4, 6.371e6 + 5.40e5, 7580, 5)
LunarReconOrbiter_rel_centre = Sat(Moon_rel_centre, "LRO", 1700.0, 50e3, 1.60e3, 10)
ISS = Sat(Earth, "ISS", 4.2e5, 4.08e5, 7.66e3, 40)


# Pre Loop Variables
bodies = []
recalculate_count = 0
current_time = 0
move_offset_x = 0
move_offset_y = 0
time_before_pause = 0
clicked_body = None
current_dt = 0
presenter_clock = 0
box = None
original_pixel_scale = 0
add_name_string = ""
add_mass_string = ""
add_radius_string = ""
add_object_radius_string = ""
add_velocity_string = ""


# Booleans
simulation_running = True
adding_body = False
calculate_next_frame = False
trajectory_tracking = True
body_locked = False
shifting = False
system_selected = False
potentials_on = False
paused = True
help_screen_visable = True
presenter_mode = False


# Run sim
if __name__ == "__main__":

    while simulation_running:

        # Set up, Which System do you want?
        while not system_selected:
            CENTRE_X = SCREEN_WIDTH // 2
            CENTRE_Y = SCREEN_HEIGHT // 2
            current_time = 0
            screen.fill(BLACK)
            select_system_menu()

            for event in pygame.event.get():

                # Quit screen
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()

                    # Check for which box you are clicking
                    system_box = which_system_box(mouse_pos)

                    if system_box == 1:
                        print("Loading System 1")
                        bodies = [
                            Sun,
                            Mercury, Venus, Earth, Moon, Mars, Phobos, Deimos,
                            Jupiter, Io, Europa, Ganymede, Callisto,
                            Saturn, Titan, Enceladus,
                            Uranus, Titania, Oberon,
                            Neptune, Triton,
                            Pluto, Charon, Ceres, Psyche, Ida, Vesta, Eris, Haumea, Makemake,
                            Halley, HaleBopp, CAPSTONE, ARTEMISP1, Hubble, ISS, Cassini, Juno,
                            Titan_LowOrbit_Sat, Dactyl, Dawn_Vesta]
                        system_selected = True
                    elif system_box == 2:
                        print("Loading System 2")
                        clicked_body = Earth_rel_centre
                        bodies = [Earth_rel_centre, Moon_rel_centre, Hubble_rel_centre, LunarReconOrbiter_rel_centre,
                                  CAPSTONE_rel_centre, ARTEMISP1_rel_centre]
                        system_selected = True
                    elif system_box == 3:
                        print("Loading System 3")
                        bodies = []
                        system_selected = True
            system_box = 0
            pygame.display.flip()

        # Enforce 60 fps
        clock.tick(FPS)

        # Event Handling

        # Get Events
        for event in pygame.event.get():

            # Quit sim
            if event.type == pygame.QUIT:
                simulation_running = False

            # Mouse Click
            if event.type == pygame.MOUSEBUTTONDOWN:
                clicked_pos = pygame.mouse.get_pos()
                if adding_body:
                    # Check where user is entering the data
                    box = which_input_box(clicked_pos)
                # we have clicked so check if we clicked a planet
                if not adding_body:
                    for body in bodies:
                        # Check for click
                        clicked_body = body.check_for_mouse_click(clicked_pos)
                        if clicked_body is not None:
                            # exit the loop and clicked body now has the body
                            break

                # If clicked body is still None, no body was clicked
                if clicked_body is None:
                    body_locked = False
                else:
                    # If a body is clicked and if the user is pressing shift
                    if shifting:
                        body_locked = True
                        move_offset_x = 0
                        move_offset_y = 0

            # Keyboard events
            if event.type == pygame.KEYDOWN:
                # Time warping
                if event.key == pygame.K_LEFTBRACKET:
                    # Insure time interval is never < base tick
                    if dt == BASETICK:
                        dt = BASETICK
                    else:
                        dt //= 2
                if event.key == pygame.K_RIGHTBRACKET:
                    dt *= 2

                # Trajectories
                if event.key == pygame.K_BACKSLASH:
                    trajectory_tracking = not trajectory_tracking

                # Zooming
                if event.key == pygame.K_EQUALS:
                    pixel_scale *= 2

                if event.key == pygame.K_MINUS:
                    pixel_scale /= 2

                # Reset sim view
                if event.key == pygame.K_TAB:
                    clicked_body = None
                    move_offset_x = 0
                    move_offset_y = 0
                    CENTRE_X = SCREEN_WIDTH // 2
                    CENTRE_Y = SCREEN_HEIGHT // 2

                # Arrow Keys move around the simulation
                if event.key == pygame.K_UP:
                    move_offset_y += 50
                if event.key == pygame.K_DOWN:
                    move_offset_y -= 50
                if event.key == pygame.K_LEFT:
                    move_offset_x += 50
                if event.key == pygame.K_RIGHT:
                    move_offset_x -= 50

                # Shifting
                if event.key == pygame.K_LSHIFT:
                    shifting = True

                # Hide/Show Help screen
                if event.key == pygame.K_PERIOD:
                    if not adding_body:
                        help_screen_visable = not help_screen_visable

                # Pausing
                if event.key == pygame.K_SPACE:
                    if not adding_body:
                        paused = not paused

                if event.key == pygame.K_COMMA and not adding_body:
                    potentials_on = not potentials_on

                # Quit
                if event.key == pygame.K_ESCAPE:
                    system_selected = False

                # Presenter Mode
                if event.key == pygame.K_p:
                    if not adding_body:
                        presenter_mode = not presenter_mode

                # Adding something to the sim
                if event.key == pygame.K_SEMICOLON:
                    adding_body = not adding_body
                    # if deToggled
                    if not adding_body:
                        box = None

                # Typing
                if adding_body:

                    if event.key == pygame.K_BACKSPACE:
                        if box == 1:
                            add_radius_string = add_radius_string[:-1]
                        if box == 2:
                            add_velocity_string = add_velocity_string[:-1]
                        if box == 3:
                            add_object_radius_string = add_object_radius_string[:-1]
                        if box == 4:
                            add_name_string = add_name_string[:-1]
                        if box == 5:
                            add_mass_string = add_mass_string[:-1]
                    else:
                        if box == 1:
                            add_radius_string += event.unicode
                        if box == 2:
                            add_velocity_string += event.unicode
                        if box == 3:
                            add_object_radius_string += event.unicode
                        if box == 4:
                            add_name_string += event.unicode
                        if box == 5:
                            add_mass_string += event.unicode

                    if event.key == pygame.K_RETURN:
                        instantiate_body(clicked_body)
                        adding_body = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT:
                    shifting = False

        # Before Simulation does any calculations fill it with Black
        screen.fill(BLACK)

        # Help Screen
        help_screen(help_screen_visable)

        # Lock Planets
        lock_planet(clicked_body, body_locked)

        # Physics
        if not paused and not help_screen_visable:
            update_simulation(dt, bodies)

        # DRAWING

        if not help_screen_visable:
            gravitational_potential_wells(bodies, potentials_on)

        # Draw the objects, only draw sats if they or their parent is clicked
        if not help_screen_visable:
            for body in bodies:

                # Always draw planets/stars
                if type(body) is Body:
                    body.draw_body()
                    body.draw_trajectories(trajectory_tracking)
                    continue

                # Below this point: body is a Sat
                if clicked_body is None:
                    continue

                # CASE 1: Clicked itself
                if body == clicked_body:
                    body.draw_body()
                    body.draw_trajectories(trajectory_tracking)
                    continue

                # CASE 2: First-level sats (normal case)
                if body.parent_body == clicked_body:
                    body.draw_body()
                    body.draw_trajectories(trajectory_tracking)
                    continue

                # If clicked is not a sat, we are done
                if type(clicked_body) is not Sat:
                    continue

                parent = clicked_body.parent_body

                # CASE 3: You clicked a sat-of-a-sat → draw parent sat
                if body == parent:
                    body.draw_body()
                    body.draw_trajectories(trajectory_tracking)
                    continue


                # CASE 4: Draw grandparent siblings (sats that share the same parent sat)
                if body.parent_body == parent:
                    body.draw_body()
                    body.draw_trajectories(trajectory_tracking)
                    continue

        # Draw data if a body is selected
        if not help_screen_visable:
            output_body_data(clicked_body)

        # Draw SYS clock
        if not help_screen_visable:
            SystemClock(current_time, dt)

        # Manage Pausing/Unpausing
        if not help_screen_visable:
            draw_pause_button(paused)

        # If paused, don't advance simulation
        if paused:
            current_time += 0
        else:
            current_time += dt

        # Add body (Doing it here so it's over the screen and bodies)
        add_body_mode(adding_body, clicked_body)

        # Presenter Mode, Iterate through bodies randomly, controlling how big it is on the screen
        if presenter_mode and presenter_clock >= (FPS * 10) and not paused:
            presenter_clock = 0
            clicked_body = presentation_mode()
            body_locked = True
        if not paused and presenter_mode:
            control_presenter_radius(clicked_body)

        presenter_clock += 1
        pygame.display.flip()

