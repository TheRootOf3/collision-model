import pygame
import time
import random
import math
import numpy as np
import itertools

multiplier = 4

# DIM_BOARD_X = 224*multiplier
# DIM_BOARD_Y = 112*multiplier
# DIM_BALL_R = math.ceil(2.8*multiplier)

DIM_BOARD_X = 800
DIM_BOARD_Y = 400
DIM_BALL_R = 20

pygame.init()
screen = pygame.display.set_mode([DIM_BOARD_X, DIM_BOARD_Y])
screen.fill((255,255,255))


class Table():
    def __init__(self):
        self.balls = []
        # self.friction = 0.9993  # 1 = no friction 
        # self.friction = 0.9999
        self.friction = 1
        self.t_time = 0

    def add_ball(self, ballID, x, y, r, vel_vector = np.array([1., 1.]), color = (255, 0, 0)):
        self.balls.append(Ball(ballID, x, y, r, vel_vector, color))
    
    def update_table(self):
        self.predict_collisions_wall(10)
        self.predict_collisions_ball(10)

        self.check_ball_collisions()
        self.apply_friction()


        for ball in self.balls:
            ball.check_wall_collisions()
            ball.update_ball(self.t_time)



    def predict_collisions_wall(self, depth_time):
        for ball in self.balls:
            col_time, wall = self.calculate_wall_col_time(ball)
            # self.set_ball_color(ball, wall)
            # print(self.return_new_wall_vector(ball, vector))
        # time.sleep(0.1)
        # time.sleep(1)
    
    def set_ball_color(self, ball, wall):
        if wall == 0:
            ball.color = (0,255,0)
        elif wall == 1:
            ball.color = (255,255,0)
        elif wall == 2:
            ball.color = (255,0,255)
        elif wall == 3:
            ball.color = (0,255,255)

    def return_new_wall_vector(self, ball, wall):
        if wall < 2:
            return (ball.vel_vector * np.array([-1.,1.]))
        else:
            return (ball.vel_vector * np.array([1.,-1.]))

    def predict_collisions_ball(self, depth_time):
        col_list = []
        for ball1, ball2 in itertools.combinations(self.balls, 2):
            col_time = self.calculate_ball_col_time(ball1, ball2)
            if col_time != None:
                col_list.append((ball1, ball2, col_time))
        
        for ball in self.balls:
            ball.color = (50,50,50)

        for ball1, ball2, _ in col_list:
            ball1.color = (255,0,0)
            ball2.color = (255,0,0)
        # time.sleep(0.01)

    def calculate_ball_col_time(self, ball1, ball2):
        c1 = np.array([ball1.x, ball1.y])
        c2 = np.array([ball2.x, ball2.y])
        
        p = [np.dot((ball1.vel_vector - ball2.vel_vector), (ball1.vel_vector - ball2.vel_vector)),
            2*np.dot(ball1.vel_vector - ball2.vel_vector, c1 - c2),
            np.dot(c1 - c2, c1 - c2) - (ball1.r + ball2.r)**2
        ]

        roots = np.roots(p)
        correct_roots = []
        for x in roots:
            if self.check_calculated_time(x, ball1) and self.check_calculated_time(x, ball2):
                correct_roots.append(x)

        # print(roots)
        # print(correct_roots)
        if len(correct_roots) == 0:
            return None
        else:
            return min(correct_roots)

    def check_calculated_time(self, time, ball):
        if time < 0 or isinstance(time, complex):
            return False
        if ball.x + time*ball.vel_vector[0] >= 0 + ball.r and ball.x + time*ball.vel_vector[0] <= DIM_BOARD_X - ball.r and\
            ball.y + time*ball.vel_vector[1] >= 0 + ball.r and ball.y + time*ball.vel_vector[1] <= DIM_BOARD_Y - ball.r:
            return True
        return False


    def calculate_wall_col_time(self, ball):
        times = [1000000, 1000000, 1000000, 1000000]
        if ball.vel_vector[0] != 0:
            times[0] = (0 + ball.r - ball.x) / ball.vel_vector[0]
            times[1] = (DIM_BOARD_X - ball.r - ball.x) / ball.vel_vector[0]
        
        if ball.vel_vector[1] != 0:
            times[2] = (0 + ball.r - ball.y) / ball.vel_vector[1]
            times[3] = (DIM_BOARD_Y - ball.r - ball.y) / ball.vel_vector[1]

        col_time = 1000000
        wall = None # wall 0 - left, wall 1 - right, wall 2 - bottom, wall 3 - top
        for i in range(len(times)):
            if times[i] < col_time and times[i] >= 0:
                col_time = times[i]
                wall = i
        
        # print(self.t_time, ball.ballID, col_time, wall)
        
        return (col_time, wall)

        
    def generate_cords(self):
        r = random.randint(20, 60)
        x = random.randint(r, DIM_BOARD_X - r)
        y = random.randint(r, DIM_BOARD_Y - r)

        return (x, y, r)

    def new_ball_cords_good(self, cords):
        for ball in self.balls:
            if (ball.x - cords[0]) ** 2 + (ball.y - cords[1]) ** 2 <= (ball.r + cords[2]) ** 2:
                return False
        
        return True
    
    def generate_random_ball(self, ball_num):
        for ballID in range (ball_num):
            cords = self.generate_cords()

            while not self.new_ball_cords_good(cords):
                cords = self.generate_cords()

            self.balls.append(Ball(
                ballID,
                cords[0],
                cords[1],
                cords[2],
                np.array([random.uniform(-1, 1), random.uniform(-1, 1)]),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                ))

    def check_ball_collisions(self):
        for ball1, ball2 in itertools.combinations(self.balls, 2):
            if (ball2.x is not ball1.x and ball2.y is not ball1.y) and (ball1.x - ball2.x) ** 2 + (ball1.y - ball2.y) ** 2 <= (ball1.r + ball2.r) ** 2:
                ball1.path.append((ball1.x, ball1.y))
                ball2.path.append((ball2.x, ball2.y))
                ball1.collisions.append((int((ball1.x + ball2.x)/2), int((ball1.y + ball2.y)/2)))
                ball2.collisions.append((int((ball1.x + ball2.x)/2), int((ball1.y + ball2.y)/2)))
                ball1.vel_vector, ball2.vel_vector = self.calculate_new_vectors(ball1.mass, ball2.mass, ball1.vel_vector, ball2.vel_vector, np.array([ball1.x, ball1.y]), np.array([ball2.x, ball2.y]))
            

    def calculate_new_vectors(self, m1, m2, v1, v2, x1, x2):
        v_new1 = v1 - 2 * m2/(m1 + m2) * np.dot(v1 - v2, x1 - x2)/((np.linalg.norm(x1 - x2)) ** 2)*(x1 - x2)
        v_new2 = v2 - 2 * m1/(m1 + m2) * np.dot(v2 - v1, x2 - x1)/((np.linalg.norm(x2 - x1)) ** 2)*(x2 - x1)
        return (v_new1, v_new2)

    def apply_friction(self):
        for ball in self.balls:
            ball.vel_vector *= self.friction 
            # if abs(ball.vel_vector[0]) < 0.0002 and abs(ball.vel_vector[1]) < 0.0002:
            #     ball.vel_vector[0] = 0
            #     ball.vel_vector[1] = 0

    def balls_stopped(self):
        for ball in self.balls:
            if abs(ball.vel_vector[0]) > 0.0006 or abs(ball.vel_vector[1]) > 0.0006:
                return False
            
        return True


class Ball():
    def __init__(self, ballID, x, y, r, vel_vector, color):
        self.ballID = ballID
        self.x = x
        self.y = y
        self.r = r
        self.mass = r
        self.color = color
        self.vel_multiplier = 2
        self.vel_vector = self.vel_multiplier * vel_vector #vel for tick
        self.path = [(x, y)]
        self.collisions = []
        self.predicted_collisions = []
    
    def update_ball(self, t_time):
        # self.x += t_time * self.vel_vector[0]
        # self.y += t_time * self.vel_vector[1]

        self.x += self.vel_vector[0]
        self.y += self.vel_vector[1]

    def check_wall_collisions(self):
        if self.x >= (DIM_BOARD_X - self.r) or self.x <= self.r:
            self.vel_vector[0] = -1 * self.vel_vector[0]
            self.path.append((self.x, self.y))
            
        if self.y >= (DIM_BOARD_Y - self.r) or self.y <= self.r:
            self.vel_vector[1] = -1 * self.vel_vector[1]
            self.path.append((self.x, self.y))

    
        


def render(screen, table):
    screen.fill((255,255,255))
    pygame.draw.line(screen, (0,255,255), (0,0), (DIM_BOARD_X,0), 5)
    pygame.draw.line(screen, (0,255,0), (0,0), (0,DIM_BOARD_Y), 5)
    pygame.draw.line(screen, (255,0,255), (0,DIM_BOARD_Y-1), (DIM_BOARD_X,DIM_BOARD_Y-1), 5)
    pygame.draw.line(screen, (255,255,0), (DIM_BOARD_X-1,0), (DIM_BOARD_X-1,DIM_BOARD_Y), 5)



    for ball in table.balls:
        # print(ball.x, ball.y)
        pygame.draw.circle(screen, ball.color, [ball.x, DIM_BOARD_Y - ball.y], ball.r)

    # for ball in table.balls[0:1]:
    #     local_path = ball.path.copy()
    #     local_path.append((ball.x, ball.y))
    #     for i in range(len(local_path) - 1):
    #         pygame.draw.line(screen, (0,0,0), (local_path[i][0], DIM_BOARD_Y - local_path[i][1]), (local_path[i+1][0], DIM_BOARD_Y - local_path[i+1][1]))

    # for ball in table.balls[0:1]:
    #     for collision in ball.collisions:
    #         pygame.draw.circle(screen, (0, 0, 255), (collision[0], DIM_BOARD_Y - collision[1]), 5)


clock = pygame.time.Clock()

table = Table()

table.generate_random_ball(5)
# table.add_ball(100*multiplier, 200/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.,0.]))
# table.add_ball(424/4*multiplier, 212/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.,-0.]))
# table.add_ball(424/4*multiplier, 188/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([-0.,0.]))

# table.add_ball(100/4*multiplier, 200/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.5,0.01]))

# table.add_ball(1, 100, 200 + DIM_BALL_R, DIM_BALL_R, np.array([1.,0.1]), (255,0,0))
# table.add_ball(2, 500, 200 + DIM_BALL_R, DIM_BALL_R, np.array([-1.,0.]), (0,120,0))


# table.add_ball(500, 100 + DIM_BALL_R, DIM_BALL_R, np.array([-1.,1.]))
# table.add_ball(100, 100 + DIM_BALL_R, DIM_BALL_R, np.array([1.,1.]))

def save_image(it_num):
    filename = str(it_num)+".png"
    pygame.image.save(screen, filename)

# time.sleep(2)
# table.t_time = 
fps = 60
while not table.balls_stopped():
    # print(table.balls_stopped())
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False   
        if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            if fps == 600:
                fps = 60
            else:
                fps = 600
    # for ball in table.balls:
    #     print(str(ball.vel_vector))
    # print("\n")
    # print(table.balls[0].vel_vector)
    table.t_time+=1
    table.update_table()
    render(screen, table)
    pygame.display.flip()
    pygame.display.set_caption(str(int(clock.get_fps())))
    # time.sleep(10)
    clock.tick(fps)

# render(screen, table)
# save_image(2)
# time.sleep(10)

