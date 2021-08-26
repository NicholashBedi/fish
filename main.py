
import pygame
import math
import numpy as np
pygame.init()

WIDTH = 500
HEIGHT = 500
FISH_WIDTH = 20
FISH_HIGHT = 20

# Set up the drawing window
screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()

pygame.display.set_caption("Boids")
icon = pygame.image.load('fish.png')

pygame.display.set_icon(icon)
fish_image = pygame.transform.scale(icon, (FISH_WIDTH,FISH_HIGHT))
n_fish = 20

DEBUG = True

fish = {"image":fish_image,
        "position": np.random.uniform(0, WIDTH, (n_fish, 2)),
        "velocity": np.random.uniform(-0.05, 0.05, (n_fish,2)),
        "acceleration": np.zeros((n_fish,2)),
        }

def draw_fishes(fish):
    for i in range(n_fish):
        orientation = math.atan2(-fish["velocity"][i][1], fish["velocity"][i][0]) * 180/ math.pi
        fish_image = pygame.transform.rotate(fish["image"], orientation)
        screen.blit(fish_image, (fish["position"][i,:]))

def wrap_around_screen(fish):
    fish["position"][fish["position"][:,0] > WIDTH, 0] -= WIDTH
    fish["position"][fish["position"][:,0] < 0, 0] += WIDTH
    fish["position"][fish["position"][:,1] > HEIGHT, 1] -= HEIGHT
    fish["position"][fish["position"][:,1] < 0, 1] += HEIGHT
    return fish

def angle_difference(a1, a2):
    diff = (a2 - a1 + 180) % 360 - 180
    if diff < -180:
        return diff + 360
    return diff

def wrap_orientation(angle):
    if angle < 0:
        angle += 360
    elif angle > 360:
        angle -= 360
    if angle < 0 or angle > 360:
        return wrap_orientation(angle)
    return angle

def seperation(fish):
    pos = fish["position"].T
    sep_matrix = pos[:, :,  np.newaxis] - pos[: ,  np.newaxis , :]
    square_distances = np.sum(sep_matrix * sep_matrix, 0)
    alert_distance_squared = 100*100
    far_fish = square_distances > alert_distance_squared

    # sep_matrix[:, destination, origin]
    for i in range(n_fish):
        start_pos = fish["position"][i,:] + [FISH_WIDTH/2, FISH_HIGHT/2]
        avoidance_force = np.zeros(2)
        n_close = 0
        for j in range(n_fish):
            if i!=j and (not far_fish[i,j]):
                if j > i and DEBUG:
                    end_pos = start_pos + sep_matrix[:,j, i]
                    pygame.draw.line(screen, (0,0,255), start_pos, end_pos, width = 2)
                n_close += 1
                force = 0.001 / (np.sqrt(square_distances[i,j]) + 0.01)
                avoidance_force += force*sep_matrix[:,i, j] / np.linalg.norm(sep_matrix[:,i, j])

        if n_close != 0:
            force_vector = avoidance_force
            end_force_vector = start_pos + 1000000 * force_vector
            if DEBUG:
                pygame.draw.line(screen, (255,0,0), start_pos, end_force_vector, width = 2)
            fish["acceleration"][i] += force_vector


def move_fish(fish, dt):
    fish["position"] += fish["velocity"] * dt
    fish["velocity"] += fish["acceleration"] * dt
    fish["velocity"] = np.clip(fish["velocity"], -0.05, 0.05)
    fish["acceleration"] = np.zeros((n_fish,2))
    return wrap_around_screen(fish)

# Run until the user asks to quit
running = True
while running:
    screen.fill((255, 255, 255))
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Draw a solid blue circle in the center
    draw_fishes(fish)
    seperation(fish)
    dt = clock.tick(60)
    fish = move_fish(fish, dt)
    # print(fish)
    pygame.display.update()

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
