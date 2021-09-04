import cv2 as cv
import numpy as np
import math
import sys
import supporting_math as sm
WIDTH = 600
HEIGHT = 600
VISION_RADIUS = 35
DEBUG = False
class Boids:
    def __init__(self, fish_image, n_fish):
        self.fish_img = fish_image
        self.img_scale = 0.25
        self.fish_img_height, self.fish_img_width = self.fish_img.shape[:2]
        self.n_fish = n_fish
        self.position = np.random.uniform(295, 305, (n_fish, 2))
        self.velocity = np.random.uniform(-0.05, 0.05, (n_fish,2))
        self.acceleration = np.zeros((n_fish,2))
        self.base_image = np.full([WIDTH,HEIGHT, 4], 255, dtype=np.uint8)
        self.time_step = 100
        self.debug_start = []
        self.debug_end = []
        self.trackbar_window = "Trackbar"
        self.make_trackbars()
        self.load_obstacle()

    def load_obstacle(self):
        self.obstacle_img = cv.imread("obstacle_1.png", cv.IMREAD_UNCHANGED)
        self.obstacle = self.obstacle_img[:, :, 0] == 0

    def make_trackbars(self):
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("cohesion", self.trackbar_window, 200, 1000, self.on_trackbar)
        cv.createTrackbar("alignment", self.trackbar_window, 200, 1000, self.on_trackbar)
        cv.createTrackbar("seperation", self.trackbar_window, 75, 1000, self.on_trackbar)
        cv.createTrackbar("avoid_wall", self.trackbar_window, 100, 1000, self.on_trackbar)
        cv.createTrackbar("avoid_object", self.trackbar_window, 70, 1000, self.on_trackbar)
        cv.createTrackbar("wall_vision", self.trackbar_window, 100, 1000, self.on_trackbar)

    def on_trackbar(self, x):
        pass

    def move_fish(self):
        max_speed = 0.08
        self.position += self.velocity * self.time_step
        self.velocity += self.acceleration * self.time_step
        self.acceleration = np.zeros((self.n_fish,2))
        for i in range(self.n_fish):
            if (True in (self.velocity[i] > max_speed)) or (True in (self.velocity[i] < -max_speed)):
                self.velocity[i] /= np.linalg.norm(self.velocity[i])
                self.velocity[i] *= max_speed

    def rotate_image(self, angle, scale = 0.25):
        image_center = tuple(np.array((self.fish_img_width, self.fish_img_height)) / 2)
        rot_mat = cv.getRotationMatrix2D(image_center, angle, scale)
        r = np.deg2rad(angle)
        new_width = scale*(abs(np.sin(r)*self.fish_img_height)
                + abs(np.cos(r)*self.fish_img_width))
        new_height = scale*(abs(np.sin(r)*self.fish_img_width)
                + abs(np.cos(r)*self.fish_img_height))
        # Translate center of image
        translate_x = (new_width-self.fish_img_width)/2
        translate_y = (new_height-self.fish_img_height)/2
        rot_mat[0,2] += translate_x
        rot_mat[1,2] += translate_y
        result = cv.warpAffine(self.fish_img, rot_mat, dsize=(int(new_width), int(new_height)))
        return result

    def display_boid(self):
        for i in range(self.n_fish):
            angle = math.atan2(-self.velocity[i][1], self.velocity[i][0]) * 180/ math.pi
            rot_fish = self.rotate_image(angle)
            y = math.floor(self.position[i][1] + 0.5)
            x = math.floor(self.position[i][0] + 0.5)
            # cv.circle(self.base_image,(x,y), 10, (0,0,255,255), 10)
            fish_part = rot_fish[:,:, 3] != 0
            y_start = y - math.floor(rot_fish.shape[0]/2)
            y_end = y + math.ceil( rot_fish.shape[0]/2)
            x_start = x - math.floor(rot_fish.shape[1]/2)
            x_end = x + math.ceil( rot_fish.shape[1]/2)
            if (y_start > 0 and y_end < HEIGHT and x_start > 0 and x_end < WIDTH):
                self.base_image[y - math.floor(rot_fish.shape[0]/2):
                           y + math.ceil( rot_fish.shape[0]/2),
                           x - math.floor(rot_fish.shape[1]/2):
                           x + math.ceil( rot_fish.shape[1]/2)][fish_part] = rot_fish[fish_part]
                self.base_image[self.base_image[:,:,3] == 0] = [255,255,255,255]
        if DEBUG:
            for i in range(self.n_fish):
                for j in range(i+1, self.n_fish):
                    if i!=j and (not self.far_fish[i,j]):
                        cv.line(self.base_image,
                            (math.floor(self.position[i][0] + 0.5),
                                math.floor(self.position[i][1] + 0.5)),
                            (math.floor(self.position[j][0] + 0.5),
                                math.floor(self.position[j][1] + 0.5)),
                            (255,0,0,255), 1)
            self.debug_start = []
            self.debug_end = []
        cv.imshow("Display window", self.base_image)
        self.base_image[:,:,3] = 0
        self.base_image[self.obstacle] = [0,0,0,255]

    def get_fish_seperations(self):
        pos = self.position.T
        self.sep_matrix = pos[:, :,  np.newaxis] - pos[: ,  np.newaxis , :]
        self.square_distances = np.sum(self.sep_matrix * self.sep_matrix, 0)
        self.alert_distance_squared = VISION_RADIUS**2
        self.far_fish = self.square_distances > self.alert_distance_squared

    def boid_behaviour(self):
        cohesion_mag = cv.getTrackbarPos("cohesion", self.trackbar_window) / 100
        alignment_mag = cv.getTrackbarPos("alignment", self.trackbar_window) / 100
        seperation_mag = cv.getTrackbarPos("seperation", self.trackbar_window) / 100
        self.get_fish_seperations()
        for i in range(self.n_fish):
            start_pos = self.position[i,:] + [self.fish_img_width/2,
                                            self.fish_img_height/2]
            avoidance_force = np.zeros(2)
            alignment_direction = np.zeros(2)
            local_flock_avg_pos = np.zeros(2)
            n_close = 0
            for j in range(self.n_fish):
                if i!=j and (not self.far_fish[i,j]):
                    if j > i and DEBUG:
                        end_pos = start_pos + self.sep_matrix[:,j, i]
                        self.debug_start.append(start_pos)
                        self.debug_end.append(end_pos)

                    n_close += 1
                    force = 0.001 / (np.sqrt(self.square_distances[i,j]) + 0.01)
                    avoidance_force += force*self.sep_matrix[:,i, j] \
                                        / np.linalg.norm(self.sep_matrix[:,i, j])
                    alignment_direction += self.velocity[j] \
                                        / np.linalg.norm(self.velocity[j])
                    local_flock_avg_pos += self.position[j]
            if n_close != 0:
                force_vector = avoidance_force
                if DEBUG:
                    end_force_vector = start_pos + 1000000 * force_vector
                    # self.debug_start.append(start_pos)
                    # self.debug_end.append(end_pos)
                self.acceleration[i] += seperation_mag * force_vector
                self.acceleration[i] += 0.0001 * alignment_mag * alignment_direction \
                                        / n_close
                local_flock_avg_pos /= n_close
                cohesion_force = 0.000002 * cohesion_mag \
                                * (local_flock_avg_pos - self.position[i,:])
                self.acceleration[i] += cohesion_force
        # self.acceleration += self.avoid_walls()

    def within_bounds(self, x, y):
        new_x = max(min(round(x), WIDTH), 0)
        new_y = max(min(round(y), HEIGHT), 0)
        return(new_x, new_y)
    def check_in_bounds(self,x,y):
        return (x>0 and x<WIDTH and y>0 and y <HEIGHT)
    def vision_cone(self):
        # draw obstacle
        # cv.rectangle(self.base_image, (WIDTH//2 - 25, HEIGHT//2 - 25),
        #             (WIDTH//2 + 25, HEIGHT//2 + 25), (0,254,0,255), 10)
        force_scale = cv.getTrackbarPos("avoid_object", self.trackbar_window) / 1000000
        blind_spot_angle = 45
        n_lines = 11
        ang_inc = (360-blind_spot_angle)/n_lines
        for i in range(self.n_fish):
            start_pos = tuple(self.position[i].astype(int))
            # Already calculated in display_boid
            angle = math.atan2(-self.velocity[i][1], self.velocity[i][0]) * 180/ math.pi
            starting_angle = sm.wrap_orientation(angle - 180 + blind_spot_angle/2)
            resulting_vector = np.zeros(2)
            n_coloisions = 0
            for j in range(n_lines):
                vision_angle = sm.wrap_orientation(starting_angle + j*ang_inc)
                end_x = start_pos[0] + VISION_RADIUS*math.cos(math.radians(vision_angle))
                end_y = start_pos[1] - VISION_RADIUS*math.sin(math.radians(vision_angle))
                end_x, end_y = self.within_bounds(end_x, end_y)
                line_points = sm.get_line_points(start_pos[0],start_pos[1], end_x, end_y)
                collision = False
                for point in line_points:
                    if self.check_in_bounds(point[0],point[1]) and self.obstacle[point[1], point[0]]:
                        collision = True
                        break
                colour = (255,0,0,255)
                if collision:
                    colour = (0,0,255,255)
                    resulting_vector += [end_x - start_pos[0], end_y - start_pos[1]]
                    n_coloisions += 1
                if DEBUG:
                    cv.line(self.base_image,start_pos,(end_x,end_y),colour,1)
            # resultant force
            if n_coloisions == n_lines:
                self.acceleration[i] = 0
                direction = [WIDTH/2, HEIGHT/2] - self.position[i]
                direction = direction/np.linalg.norm(direction)
                self.velocity[i] = 0.08 * direction
            if n_coloisions > 0:
                resulting_vector /= -VISION_RADIUS
                if DEBUG:
                    end_r_vector = tuple((VISION_RADIUS*resulting_vector + [start_pos[0], start_pos[1]]).astype(int))
                    cv.line(self.base_image,start_pos,end_r_vector,(255,0,255,255),4)
                self.acceleration[i] /= 2
                self.acceleration[i] += force_scale*resulting_vector

    def check_boundaries(self):
        too_right = self.position[:, 0] > WIDTH - self.fish_img_width
        too_left = self.position[:, 0] < self.fish_img_width
        too_low = self.position[:, 1] > HEIGHT - self.fish_img_height
        too_high = self.position[:, 1] < self.fish_img_height

        self.position[too_right, 0] = WIDTH - self.fish_img_width
        self.velocity[too_right, 0] = -1*(abs(self.velocity[too_right, 0]))

        self.position[too_left, 0] = self.fish_img_width
        self.velocity[too_left, 0] = abs(self.velocity[too_left, 0])

        self.position[too_low, 1] = HEIGHT - self.fish_img_height
        self.velocity[too_low, 1] = -1*(abs(self.velocity[too_low, 1]))

        self.position[too_high, 1] = self.fish_img_height
        self.velocity[too_high, 1] = abs(self.velocity[too_high, 1])

    def avoid_walls(self):
        too_close = cv.getTrackbarPos("wall_vision", self.trackbar_window)
        close_to_right_wall = self.position[:, 0] > WIDTH - too_close
        close_to_left_wall = self.position[:, 0] < too_close
        close_to_bottom_wall= self.position[:, 1] > HEIGHT - too_close
        close_to_top = self.position[:, 1] < too_close
        avoid_walls_force = np.zeros((self.n_fish,2))
        force_scale = cv.getTrackbarPos("avoid_wall", self.trackbar_window) / 10000
        # force_scale = 1000
        avoid_divide_by_zero = 0.5
        avoid_walls_force[close_to_right_wall, 0] = -force_scale / (WIDTH - self.position[close_to_right_wall, 0] + avoid_divide_by_zero)
        avoid_walls_force[close_to_left_wall, 0] = force_scale / (self.position[close_to_left_wall, 0] + avoid_divide_by_zero)
        avoid_walls_force[close_to_bottom_wall, 1] = -force_scale / (HEIGHT - self.position[close_to_bottom_wall, 1] + avoid_divide_by_zero)
        avoid_walls_force[close_to_top, 1] = force_scale / (self.position[close_to_top, 1] + avoid_divide_by_zero)
        return avoid_walls_force

fish_img = cv.imread("fish.png", cv.IMREAD_UNCHANGED)
b = Boids(fish_img, 60)
while(True):
    b.boid_behaviour()
    b.vision_cone()
    b.move_fish()
    b.display_boid()
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
