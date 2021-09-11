import cv2 as cv
import numpy as np
import math
import sys
import supporting_math as sm
num_fish = 60
WIDTH = 600
HEIGHT = 600
DEBUG = False
time_delay = 1
WRITE_VIDEO = True
video_name = "test2.avi"
# obstacle_img_name = "test_5050.png"
# obstacle_img_name = "test_left.png"
# obstacle_img_name = "test_right.png"
obstacle_img_name = "obstacle_1.png"
# TEST = True
TEST = False
class Boids:
    def __init__(self, fish_image, n_fish, test = False):
        self.fish_img = fish_image
        self.img_scale = 0.25
        self.fish_img_height, self.fish_img_width = self.fish_img.shape[:2]
        self.n_fish = n_fish
        if TEST:
            self.position = np.zeros((n_fish,2))
            self.position[:, 0] = np.full((n_fish), 150)
            self.position[:, 1] = np.full((n_fish), 300)
            self.velocity = np.zeros((n_fish,2))
            self.velocity[:, 0] = np.full((n_fish), 0.05)
        else:
            self.position = np.random.uniform(295, 305, (n_fish, 2))
            self.velocity = np.random.uniform(-0.05, 0.05, (n_fish,2))
        if WRITE_VIDEO:
            self.set_up_video_recorder()
        self.acceleration = np.zeros((n_fish,2))
        self.base_image = np.full([WIDTH,HEIGHT, 4], 255, dtype=np.uint8)
        self.time_step = 100
        self.debug_start = []
        self.debug_end = []
        self.trackbar_window = "Trackbar"
        blind_spot_angle = 45
        self.n_lines = 10
        self.ang_inc = (360-blind_spot_angle)/self.n_lines
        self.vision_radius = 50
        self.fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.make_trackbars()
        self.load_obstacle()

    def set_up_video_recorder(self):
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = 12
        self.writer = cv.VideoWriter(video_name, fourcc, 12, (WIDTH, HEIGHT))
    def load_obstacle(self):
        self.obstacle_img = cv.imread(obstacle_img_name, cv.IMREAD_UNCHANGED)
        self.obstacle = self.obstacle_img[:, :, 0] == 0

    def make_trackbars(self):
        cv.namedWindow(self.trackbar_window)
        cv.createTrackbar("cohesion", self.trackbar_window, 100, 1000, self.on_trackbar)
        cv.createTrackbar("alignment", self.trackbar_window, 100, 1000, self.on_trackbar)
        cv.createTrackbar("seperation", self.trackbar_window, 50, 1000, self.on_trackbar)
        # cv.createTrackbar("avoid_wall", self.trackbar_window, 100, 1000, self.on_trackbar)
        cv.createTrackbar("a_obj", self.trackbar_window, 100, 1000, self.on_trackbar)
        cv.createTrackbar("v_radius", self.trackbar_window, 100, 150, self.on_trackbar)

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
            rot_fish = sm.rotate_image(self.fish_img, angle, scale = 0.25)
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
        if WRITE_VIDEO:
            self.writer.write(self.base_image[:,:, :3])
        self.base_image[:,:,3] = 0
        self.base_image[self.obstacle] = [0,0,0,255]

    def get_fish_seperations(self):
        pos = self.position.T
        self.sep_matrix = pos[:, :,  np.newaxis] - pos[: ,  np.newaxis , :]
        self.square_distances = np.sum(self.sep_matrix * self.sep_matrix, 0)
        self.alert_distance_squared = self.vision_radius**2
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

    def get_end_points(self, start_pos, angle, scale = 1):
        end_x = start_pos[0] + scale*self.vision_radius*math.cos(math.radians(angle))
        end_y = start_pos[1] - scale*self.vision_radius*math.sin(math.radians(angle))
        return self.within_bounds(end_x, end_y)

    def check_colision(self, start_pos, vision_angle, scale = 1):
        end_x, end_y = self.get_end_points(start_pos, vision_angle, scale)
        line_points = sm.get_line_points(start_pos[0],start_pos[1], end_x, end_y)
        collision = False
        for point in line_points:
            if self.check_in_bounds(point[0],point[1]) and self.obstacle[point[1], point[0]]:
                return True
        return False

    def display_vision_line(self, start_pos, angle, collision):
        end_x, end_y = self.get_end_points(start_pos, angle)
        colour = (255,0,0,255)
        if collision:
            colour = (0,0,255,255)
        cv.line(self.base_image,start_pos,(end_x,end_y),colour,1)

    def choose_direction(self, start_pos, angle_cw, angle_ccw):
        max_scale = 2
        min_scale = 1
        scale = 2
        collision_ccw = self.check_colision(start_pos, angle_ccw, scale)
        collision_cw = self.check_colision(start_pos, angle_cw, scale)
        if (not collision_ccw) and collision_cw:
            return angle_ccw
        elif (not collision_cw) and collision_ccw:
            return angle_cw
        # bias to ccw
        elif not collision_ccw and not collision_cw:
            return angle_ccw
        # Will only go forward if both collisions hit
        scale = 1.5
        iter = 0
        while iter < 15:
            collision_ccw = self.check_colision(start_pos, angle_ccw, scale)
            collision_cw = self.check_colision(start_pos, angle_cw, scale)
            if (not collision_ccw) and collision_cw:
                return angle_ccw
            elif (not collision_cw) and collision_ccw:
                return angle_cw
            elif not collision_ccw and not collision_cw:
                min_scale = scale
            else:
                max_scale = scale
            scale = (max_scale - min_scale)/2 + min_scale
            iter += 1
        return angle_ccw

    def find_nearest_clear_angle(self, start_pos, starting_angle):
        for i in range(1, self.n_lines):
            angle_ccw = sm.wrap_orientation(starting_angle + i*self.ang_inc/2)
            angle_cw =  sm.wrap_orientation(starting_angle - i*self.ang_inc/2)
            collision_ccw = self.check_colision(start_pos, angle_ccw, scale = 1)
            collision_cw = self.check_colision(start_pos, angle_cw, scale = 1)
            if DEBUG:
                self.display_vision_line(start_pos, angle_ccw, collision_ccw)
                self.display_vision_line(start_pos, angle_cw, collision_cw)
            if (not collision_ccw) and collision_cw:
                return angle_ccw
            elif (not collision_cw) and collision_ccw:
                return angle_cw
            elif not collision_cw and not collision_ccw:
                return self.choose_direction(start_pos, angle_cw, angle_ccw)
        return_angle = math.atan2((WIDTH/2 - start_pos[0]), (HEIGHT/2 - start_pos[1])) * 180/ math.pi
        acceleration = []
        return return_angle

    def how_much_to_increase_angle(self, starting_angle, angle_to_move):
        diff = sm.wrap_orientation(angle_to_move - starting_angle)
        change_angle_by = 20
        if diff == 0 or diff == 180:
            return angle_to_move
        if diff > 180:
            return(sm.wrap_orientation(angle_to_move - change_angle_by))
        else:
            return(sm.wrap_orientation(angle_to_move + change_angle_by))
    def vision_cone(self):
        force_scale = cv.getTrackbarPos("a_obj", self.trackbar_window) / 10000
        self.vision_radius = cv.getTrackbarPos("v_radius", self.trackbar_window)
        for i in range(self.n_fish):
            start_pos = tuple(self.position[i].astype(int))
            # Already calculated in display_boid
            angle = math.atan2(-self.velocity[i][1], self.velocity[i][0]) * 180/ math.pi
            starting_angle = sm.wrap_orientation(angle)

            if DEBUG:
                end_x, end_y = self.get_end_points(start_pos, angle)
                cv.line(self.base_image,start_pos,(end_x,end_y),(255,0,255,255),1)
            if self.check_colision(start_pos, starting_angle):
                clear_angle = self.find_nearest_clear_angle(start_pos, starting_angle)
                angle_to_move = self.how_much_to_increase_angle(starting_angle, clear_angle)
                direction = np.array((math.cos(math.radians(angle_to_move)),
                    -1*math.sin(math.radians(angle_to_move))))
                self.acceleration[i] += force_scale \
                            *direction

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
print(fish_img.shape)
np.set_printoptions(threshold=sys.maxsize)
print(fish_img)
b = Boids(fish_img, num_fish)
while(True):
    b.boid_behaviour()
    b.vision_cone()
    b.move_fish()
    b.display_boid()
    if cv.waitKey(time_delay) & 0xFF == ord('q'):
        break
b.writer.release()
