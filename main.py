
import pygame
import math
import numpy as np
import cv2 as cv
pygame.init()

WIDTH = 500
HEIGHT = 500
FISH_WIDTH = 20
FISH_HIGHT = 20
VISION_RADIUS = 100

# Set up the drawing window
screen = pygame.display.set_mode([WIDTH, HEIGHT])
clock = pygame.time.Clock()

pygame.display.set_caption("Boids")
icon = pygame.image.load('images/fish.png')

pygame.display.set_icon(icon)
fish_image = pygame.transform.scale(icon, (FISH_WIDTH,FISH_HIGHT))
n_fish = 50
trackbar_window = "Trackbar"
DEBUG = False

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


def get_fish_seperations(fish):
    pos = fish["position"].T
    sep_matrix = pos[:, :,  np.newaxis] - pos[: ,  np.newaxis , :]
    square_distances = np.sum(sep_matrix * sep_matrix, 0)
    alert_distance_squared = VISION_RADIUS**2
    far_fish = square_distances > alert_distance_squared
    return sep_matrix, square_distances, far_fish

# def vision_cone(fish):
#     VISION_RADIUS

def avoid_walls(fish):
    too_close = 50
    close_to_right_wall = fish["position"][:, 0] > WIDTH - too_close
    close_to_left_wall = fish["position"][:, 0] < too_close
    close_to_bottom_wall= fish["position"][:, 1] > HEIGHT - too_close
    close_to_top = fish["position"][:, 1] < too_close
    avoid_walls_force = np.zeros((n_fish,2))
    force_scale = cv.getTrackbarPos("avoid_object", trackbar_window) / 100000
    # force_scale = 1000
    avoid_divide_by_zero = 0.5
    avoid_walls_force[close_to_right_wall, 0] = -force_scale / (WIDTH - fish["position"][close_to_right_wall, 0] + avoid_divide_by_zero)
    avoid_walls_force[close_to_left_wall, 0] = force_scale / (fish["position"][close_to_left_wall, 0] + avoid_divide_by_zero)
    avoid_walls_force[close_to_bottom_wall, 1] = -force_scale / (HEIGHT - fish["position"][close_to_bottom_wall, 1] + avoid_divide_by_zero)
    avoid_walls_force[close_to_top, 1] = force_scale / (fish["position"][close_to_top, 1] + avoid_divide_by_zero)
    return avoid_walls_force

def boid_behaviour(fish, sep_matrix, square_distances, far_fish, seperation_mag = 1, alignment_mag = 1, cohesion_mag = 1):
    cohesion_mag = cv.getTrackbarPos("cohesion", trackbar_window) / 100
    alignment_mag = cv.getTrackbarPos("alignment", trackbar_window) / 100
    seperation_mag = cv.getTrackbarPos("seperation", trackbar_window) / 100
    for i in range(n_fish):
        start_pos = fish["position"][i,:] + [FISH_WIDTH/2, FISH_HIGHT/2]
        avoidance_force = np.zeros(2)
        alignment_direction = np.zeros(2)
        local_flock_avg_pos = np.zeros(2)
        n_close = 0
        for j in range(n_fish):
            if i!=j and (not far_fish[i,j]):
                if j > i and DEBUG:
                    end_pos = start_pos + sep_matrix[:,j, i]
                    pygame.draw.line(screen, (0,0,255), start_pos, end_pos, width = 2)
                n_close += 1
                force = 0.001 / (np.sqrt(square_distances[i,j]) + 0.01)
                avoidance_force += force*sep_matrix[:,i, j] / np.linalg.norm(sep_matrix[:,i, j])
                alignment_direction += fish["velocity"][j]  / np.linalg.norm(fish["velocity"][j])
                local_flock_avg_pos += fish["position"][j]
        if n_close != 0:
            force_vector = avoidance_force
            if DEBUG:
                end_force_vector = start_pos + 1000000 * force_vector
                pygame.draw.line(screen, (255,0,0), start_pos, end_force_vector, width = 2)
            fish["acceleration"][i] += seperation_mag * force_vector
            fish["acceleration"][i] += 0.00001 * alignment_mag * alignment_direction / n_close
            local_flock_avg_pos /= n_close
            cohesion_force = 0.000002 * cohesion_mag * (local_flock_avg_pos - fish["position"][i,:])
            fish["acceleration"][i] += cohesion_force
    fish["acceleration"] += avoid_walls(fish)


def move_fish(fish, dt):
    max_speed = 0.06
    fish["position"] += fish["velocity"] * dt
    fish["velocity"] += fish["acceleration"] * dt
    fish["acceleration"] = np.zeros((n_fish,2))
    for i in range(n_fish):
        if (True in (fish["velocity"][i] > max_speed)) or (True in (fish["velocity"][i] < -max_speed)):
            fish["velocity"][i] /= np.linalg.norm(fish["velocity"][i])
            fish["velocity"][i] *= max_speed
    return wrap_around_screen(fish)

# Run until the user asks to quit
running = True

def on_trackbar(val):
    pass

cv.namedWindow(trackbar_window)
cv.createTrackbar("cohesion", trackbar_window, 100, 1000, on_trackbar)
cv.createTrackbar("alignment", trackbar_window, 100, 1000, on_trackbar)
cv.createTrackbar("seperation", trackbar_window, 100, 1000, on_trackbar)
cv.createTrackbar("avoid_object", trackbar_window, 200, 1000, on_trackbar)
while running:
    screen.fill((255, 255, 255))
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # Draw a solid blue circle in the center
    draw_fishes(fish)
    sep_matrix, square_distances, far_fish = get_fish_seperations(fish)
    boid_behaviour(fish, sep_matrix, square_distances, far_fish,
                seperation_mag = 0.5, alignment_mag = 3, cohesion_mag = 2)
    dt = clock.tick(6000)
    fish = move_fish(fish, dt)
    # print(fish)
    pygame.display.update()

    # Flip the display
    pygame.display.flip()
    # running = False

# Done! Time to quit.
pygame.quit()
