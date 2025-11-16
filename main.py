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

# Update physics
def update_simulation(dt: int, body_list: List[Body]) -> None:

    """
    The algorithm uses the base tick approach, the program has a minimum physics step
    and dt is the delta time per fram, the engine will calculate physics steps
    in units of base ticks
    Takes a dt variable for the engines current delta time, and calculates the physics
    per minimum time step to ensure changing delta time doesn't implement distortions
    """

    for i in range(dt // BASETICK):
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

    if locked and check_body is not None:
        CENTRE_X = int(round(-(check_body.position[0] * pixel_scale) + SCREEN_WIDTH / 2 + move_offset_x))
        CENTRE_Y = int(round(-(check_body.position[1] * pixel_scale) + SCREEN_HEIGHT / 2 + move_offset_y))

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
def SystemClock(seconds: int, dt) -> None:
    # Get the global time
    days = seconds // 86400
    seconds %= 86400

    hours = seconds // 3600
    seconds %= 3600

    minutes = seconds // 60
    seconds %= 60

    clock_str = f'{days}days {hours}hours'
    render_text(clock_str, (int(3/4 * SCREEN_WIDTH) , int(1/8 * SCREEN_HEIGHT)))
    render_text(f'{dt} seconds per frame',(int(3/4 * SCREEN_WIDTH) , int(1/8 * SCREEN_HEIGHT) + 20))

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
        render_text("KeyBinds:", (int(1/4 * SCREEN_WIDTH) + 100, int(1/8 * SCREEN_HEIGHT)))
        render_text("BACKSLASH : Show/Hide Planet Trajectories", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 20))
        render_text("Space : Pause/Play Simulation", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 40))
        render_text("[ and ] : Increase/ Decrease Time per frame", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 60))
        render_text("- and = : Zoom In/Out", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 80))
        render_text("TAB : Reset Simulation View", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 100))
        render_text("Arrow Keys : Move Around Simulation", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 120))
        render_text("PERIOD : Show/Hide Help Screen", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 140))
        render_text("Click on a Body to show its data AND Shift + Left click to focus on a body", (int(1 / 4 * SCREEN_WIDTH) + 100, int(1 / 8 * SCREEN_HEIGHT) + 160))

def presentation_mode() -> None | Body | Sat | Any:
    random_index = int(random.randint(0, len(bodies) - 1))
    body_to_return = bodies[random_index]
    return body_to_return

def control_presenter_radius(current_body: Body) -> None:
    global pixel_scale
    global original_pixel_scale
    original_pixel_scale = pixel_scale
    if clicked_body != None:
        if type(current_body) == Sat:
            parent_body_draw_size = current_body.parent_body.size * pixel_scale
            if parent_body_draw_size >= (SCREEN_WIDTH * 1/16):
                pixel_scale /= 2
            else:
                pixel_scale = original_pixel_scale
        else:
            draw_size = current_body.size * pixel_scale
            if draw_size >= (SCREEN_WIDTH * 1/16):
                pixel_scale /= 2
            else:
                pixel_scale = original_pixel_scale






# Instances

# Stars
Sun = Body("Sun", 1.9885e30, 0, 0, 6.957e8)


# PLANETS (orbital radius & velocity around Sun)
Mercury = Body("Mercury", 3.3011e23, 5.79e10, 4.79e4, 2.4397e6)
Venus = Body("Venus", 4.867e24, 1.082e11, 3.5e4, 6.0518e6)
Earth = Body("Earth", 5.97237e24, 1.496e11, 2.978e4, 6.371e6)
Mars = Body("Mars", 6.4171e23, 2.279e11, 2.41e4, 3.3895e6)
Jupiter = Body("Jupiter", 1.898e27, 7.7857e11, 1.307e4, 6.9911e7)
Saturn = Body("Saturn", 5.683e26, 1.4335e12, 9.68e3, 5.8232e7)
Uranus = Body("Uranus", 8.681e25, 2.8725e12, 6.8e3, 2.5362e7)
Neptune = Body("Neptune", 1.024e26, 4.495e12, 5.43e3, 2.4622e7)

# MOONS (orbital radius & velocity relative to parent body center)
Moon = Sat(Earth, "Moon", 7.3477e22, 3.844e8, 1.022e3, 1.7374e6)
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

# COMETS (orbit Sun)
Halley = Body("Halley's Comet", 2.2e14, 5.3e12, 5.4e4, 5500)
HaleBopp = Body("Comet Hale-Bopp", 2.2e14, 7.2e12, 4.7e4, 30000)
C67P = Body("Comet 67P", 1e13, 5.87e11, 1.5e4, 2000)

# ARTIFICIAL SATELLITES (radius from parent surface)
CAPSTONE = Sat(Moon, "CAPSTONE", 25, 1.5e6, 1.0e3, 2)
ARTEMISP1 = Sat(Moon, "ARTEMIS P1", 500, 1.8e6, 1.5e3, 2)
Cassini = Sat(Saturn, "Cassini", 2125.0, 11646e3, 23297.6, 6.7)
Juno = Sat(Jupiter, "Juno", 3625.0, 4200e3, 41342.8, 3.5)
Titan_LowOrbit_Sat = Sat(Titan, "Titan_LowOrbiter", 5.0e2, 100e3, 1831.8, 2.0)

Hubble = Sat(Earth, "Hubble Space Telescope", 11110, 6.9e6, 7.6e3, 5)
ISS = Sat(Earth, "ISS", 4.2e5, 4.08e5, 7.66e3, 40)


# Pre Loop Variables
recalculate_count = 0
current_time = 0
move_offset_x = 0
move_offset_y = 0
time_before_pause = 0
clicked_body = None
current_dt = 0
presenter_clock = 0
original_pixel_scale = 0

# Booleans
simulation_running = True
calculate_next_frame = False
trajectory_tracking = True
body_locked = False
shifting = False
paused = True
help_screen_visable = True
presenter_mode = False


# Lists
#bodies = [Earth, Moon, LunarReconOrbiter, GeoSat, CAPSTONE, ARTEMISP1]

bodies = [
    Sun,
    Mercury, Venus, Earth, Moon, Mars, Phobos, Deimos,
    Jupiter, Io, Europa, Ganymede, Callisto,
    Saturn, Titan, Enceladus,
    Uranus, Titania, Oberon,
    Neptune, Triton,
    Pluto, Charon, Ceres, Eris, Haumea, Makemake,
    Halley, HaleBopp, CAPSTONE, ARTEMISP1, Hubble, ISS, Cassini, Juno, Titan_LowOrbit_Sat,]

# Run sim
if __name__ == "__main__":

    while simulation_running:
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
                # we have clicked so check if we clicked a planet
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
                    help_screen_visable = not help_screen_visable

                # Pausing
                if event.key == pygame.K_SPACE:
                    paused = not paused

                # Quit
                if event.key == pygame.K_ESCAPE:
                    simulation_running = False

                # Presenter Mode
                if event.key == pygame.K_p:
                    presenter_mode = not presenter_mode


            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT:
                    shifting = False

        # Before Simulation does any calculations fill it with Black
        screen.fill(BLACK)

        help_screen(help_screen_visable)

        # Physics
        if not paused and not help_screen_visable:
            update_simulation(dt, bodies)

        # DRAWING

        # Draw the objects
        if not help_screen_visable:
            for body in bodies:
                #if (type(body) is Body) or (type(body) is Sat and body.parent_body == clicked_body):
                body.draw_body()
                body.draw_trajectories(trajectory_tracking)

        # Draw data if a body is selected
        if not help_screen_visable:
            output_body_data(clicked_body)

        # Lock Planets
        lock_planet(clicked_body, body_locked)

        # Draw SYS clock
        if not help_screen_visable:
            SystemClock(current_time, dt)

        # Manage Pausing
        if not help_screen_visable:
            draw_pause_button(paused)

        # Next frame
        if paused:
            current_time += 0
        else:
            current_time += dt

        # Presenter Mode
        if presenter_mode and presenter_clock >= (FPS * 10) and not paused:
            presenter_clock = 0
            clicked_body = presentation_mode()
            body_locked = True
        if not paused and presenter_mode:
            control_presenter_radius(clicked_body)


        presenter_clock += 1
        pygame.display.flip()

