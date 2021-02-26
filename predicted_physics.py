import pygame
import time
import random
import numpy as np
import itertools
import copy
import sys

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
        # self.friction = 0.93  # 1 = no friction 
        # self.friction = 0.9999
        self.friction = 1
        # self.t_time = 0
        self.collision_list = []
        self.d_time = 0

    def copy_table(self):
        newTable = Table()
        newTable.balls = copy.deepcopy(self.balls)
        newTable.friction = self.friction
        # newTable.t_time = self.t_time
        newTable.d_time = self.d_time

        
        return newTable


    def add_ball(self, ballID, pos, r, vel_vector = np.array([1., 1.]), color = (255, 0, 0)):
        self.balls.append(Ball(ballID, pos, r, vel_vector, color))
    

    def check_for_update_prediction(self, anim_time):
        if anim_time >= self.d_time:
            self.update_table()
            self.predict_all_collisions(1) #prediction depth

    def update_table(self):

        if len(table.collision_list) != 0:
            for element in table.collision_list:
                    for ball in table.balls:
                        if ball.ballID == element[0]:
                            ball.vel_vector = copy.deepcopy(element[2])
                            ball.pos = copy.deepcopy(element[3])
                            # print(ball.pos)
                            ball.ball_time = element[1]
                
        for ball in table.balls:
            # print(ball.pos, ball.vel_vector)
            ball.update_ball_pos(self.d_time- ball.ball_time)
            ball.startpos = copy.deepcopy(ball.pos)
            ball.ball_time = self.d_time
        
        # self.predict_collisions_ball()
        # self.predict_collisions_wall()

        # self.predict_all_collisions(1000)

        # time.sleep(0.2)

        # self.check_ball_collisions()
        # self.apply_friction()


        # for ball in self.balls:
        #     ball.check_wall_collisions()
        #     ball.update_ball(self.t_time)




    def predict_all_collisions(self, depth_time):
        tmp_depth_time = depth_time

        self.collision_list = []
        pred_table = self.copy_table()

        while depth_time >= 0:

            first_col = pred_table.predict_wall_ball()

            if first_col[-1] > depth_time:
                break

            
            for ball in pred_table.balls:
                ball.update_ball_pos(first_col[-1])

            if first_col[0] == 0:
                # for ball in pred_table.balls:
                #     ball.update_ball_pos(first_col[-1])
                first_col[1].vel_vector = pred_table.return_new_wall_vector(first_col[1], first_col[2])

                depth_time -= first_col[-1]
            
                self.collision_list.append([first_col[1].ballID, pred_table.d_time + first_col[-1], copy.deepcopy(first_col[1].vel_vector), copy.deepcopy(first_col[1].pos)])
                pred_table.d_time += first_col[-1]

            else:


                first_col[1].vel_vector, first_col[2].vel_vector = pred_table.calculate_new_vectors(first_col[1].mass, first_col[2].mass, first_col[1].vel_vector, first_col[2].vel_vector, first_col[1].pos, first_col[2].pos)
                
                depth_time -= first_col[-1]
                self.collision_list.append([first_col[1].ballID, pred_table.d_time + first_col[-1], copy.deepcopy(first_col[1].vel_vector), copy.deepcopy(first_col[1].pos)])
                self.collision_list.append([first_col[2].ballID, pred_table.d_time + first_col[-1], copy.deepcopy(first_col[2].vel_vector), copy.deepcopy(first_col[2].pos)])
                pred_table.d_time += first_col[-1]

        self.d_time += tmp_depth_time

    def return_min_index(self, list_of_tuples):
        if len(list_of_tuples) is 0:
            return None 

        min_val = 1000000
        min_indx = -1
        for x in range(len(list_of_tuples)):
            if list_of_tuples[x][-1] < min_val:
                min_val = list_of_tuples[x][-1]
                min_indx = x

        return min_indx
    
    def return_first_col(self, col1, col2):
        if col2 is None or col1[-1] < col2[-1]:
            return col1
        else:
            return col2

    def predict_wall_ball(self):
        col_list = []
        for ball in self.balls:
            col_time, wall = self.calculate_wall_col_time(ball)
            col_list.append((0, ball, wall, col_time))

        for ball1, ball2 in itertools.combinations(self.balls, 2):
            col_time = self.calculate_ball_col_time(ball1, ball2)
            if col_time != None:
                col_list.append((1, ball1, ball2, col_time))
        
        min_index = self.return_min_index(col_list)
        if min_index is None:
            return None 
        else:
            return col_list[min_index]
    
    def set_ball_color(self, ball, wall):
        if ball.color != (255,0,0):
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
            return (ball.vel_vector * np.array([-1,1]))
        else:
            return (ball.vel_vector * np.array([1,-1]))

    def calculate_ball_col_time(self, ball1, ball2):
       
        p = [np.dot((ball1.vel_vector - ball2.vel_vector), (ball1.vel_vector - ball2.vel_vector)),
            2*np.dot(ball1.vel_vector - ball2.vel_vector, ball1.pos - ball2.pos),
            np.linalg.norm(ball1.pos - ball2.pos)**2 - (ball1.r + ball2.r)**2
        ]

        roots = np.roots(p)
        correct_roots = []
        for x in roots:
            if self.check_calculated_time(x, ball1) and self.check_calculated_time(x, ball2):
                correct_roots.append(x)

        if len(correct_roots) == 0:
            return None
        else:
            return min(correct_roots)

    def check_calculated_time(self, time, ball):
        if time <= 0.00001 or isinstance(time, complex): #added time <= 0.00001 to get rid of two balls blocking. May cause problems!
            return False
        if ball.pos[0] + time*ball.vel_vector[0] >= 0 + ball.r and ball.pos[0] + time*ball.vel_vector[0] <= DIM_BOARD_X - ball.r and\
            ball.pos[1] + time*ball.vel_vector[1] >= 0 + ball.r and ball.pos[1] + time*ball.vel_vector[1] <= DIM_BOARD_Y - ball.r:
            return True
        return False


    def calculate_wall_col_time(self, ball):
        times = [1000000, 1000000, 1000000, 1000000]
        if ball.vel_vector[0] != 0:
            times[0] = (0 + ball.r - ball.pos[0]) / ball.vel_vector[0]
            times[1] = (DIM_BOARD_X - ball.r - ball.pos[0]) / ball.vel_vector[0]
        
        if ball.vel_vector[1] != 0:
            times[2] = (0 + ball.r - ball.pos[1]) / ball.vel_vector[1]
            times[3] = (DIM_BOARD_Y - ball.r - ball.pos[1]) / ball.vel_vector[1]

        col_time = 1000000
        wall = None # wall 0 - left, wall 1 - right, wall 2 - bottom, wall 3 - top
        for i in range(len(times)):
            if times[i] < col_time and times[i] >= 0.00001: #added time <= 0.00001 to get rid of two balls blocking. May cause problems!
                col_time = times[i]
                wall = i
        
        # print(self.t_time, ball.ballID, col_time, wall)
        
        return (col_time, wall)

        
    def generate_cords(self):
        r = random.randint(20, 60)
        pos = np.array([random.uniform(r, DIM_BOARD_X - r), random.uniform(r, DIM_BOARD_Y - r)])

        return (pos, r)

    def new_ball_cords_good(self, pos, r):
        for ball in self.balls:
            if np.linalg.norm(ball.pos - pos) <= (ball.r + r):
                return False
        
        return True
    
    def generate_random_ball(self, ball_num):
        for ballID in range (ball_num):
            pos, r = self.generate_cords()

            while not self.new_ball_cords_good(pos, r):
                pos, r = self.generate_cords()

            self.balls.append(Ball(
                ballID,
                pos,
                r,
                np.array([random.uniform(-1, 1), random.uniform(-1, 1)]),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                ))

    def check_ball_collisions(self):
        for ball1, ball2 in itertools.combinations(self.balls, 2):
            if (ball1.ballID is not ball2.ballID) and np.linalg.norm(ball1.pos - ball2.pos) <= (ball1.r + ball2.r):
                ball1.path.append(ball1.pos)
                ball2.path.append(ball2.pos)
                ball1.collisions.append((ball1.pos + ball2.pos)/2)
                ball2.collisions.append((ball1.pos + ball2.pos)/2)
                ball1.vel_vector, ball2.vel_vector = self.calculate_new_vectors(ball1.mass, ball2.mass, ball1.vel_vector, ball2.vel_vector, ball1.pos, ball2.pos)
            

    def calculate_new_vectors(self, m1, m2, v1, v2, pos1, pos2):
        v_new1 = v1 - 2 * m2/(m1 + m2) * np.dot(v1 - v2, pos1 - pos2)/((np.linalg.norm(pos1 - pos2)) ** 2)*(pos1 - pos2)
        v_new2 = v2 - 2 * m1/(m1 + m2) * np.dot(v2 - v1, pos2 - pos1)/((np.linalg.norm(pos2 - pos1)) ** 2)*(pos2 - pos1)
        return (v_new1, v_new2)

    def apply_friction(self):
        for ball in self.balls:
            ball.vel_vector *= self.friction 

    def balls_stopped(self):
        for ball in self.balls:
            if abs(ball.vel_vector[0]) > 0.0006 or abs(ball.vel_vector[1]) > 0.0006:
                return False
            
        return True

    def create_ball_set_to_render(self, t_time2):
        ball_set = []

        for ball in self.balls:
            # print(ball.ballID, ball.pos)
            ball_set.append(ball.copy_ball())

        if len(self.collision_list) > 0:

            if t_time2 > self.d_time:
                print("END!!!!")
                return None

            for element in self.collision_list:
                # print(element)
                # print(t_time)
                # time.sleep(1)
                if element[1] > t_time2:
                    break
                else:
                    for ball in ball_set:
                        if ball.ballID == element[0]:
                            ball.vel_vector = copy.deepcopy(element[2])
                            ball.pos = copy.deepcopy(element[3])
                            # print(ball.pos)
                            ball.ball_time = copy.deepcopy(element[1])
        
        for ball in ball_set:
            ball.pos[0] = ball.pos[0] + (ball.vel_vector[0] * (t_time2 - ball.ball_time))
            ball.pos[1] = ball.pos[1] + (ball.vel_vector[1] * (t_time2 - ball.ball_time))
        # for ball in self.balls:
            # print(ball.ballID, ball.pos)
        # time.sleep(10)
        # print(ball_set)
        return ball_set


class Ball():
    def __init__(self, ballID, pos, r, vel_vector, color):
        self.ballID = ballID
        self.startpos = pos
        self.pos = pos
        self.r = r
        self.mass = r ** 3
        self.color = color
        self.vel_vector = vel_vector #vel for tick
        self.path = [pos]
        self.collisions = []
        self.predicted_collisions = []
        self.ball_time = 0

    def copy_ball(self):
        newBall = Ball(self.ballID, copy.deepcopy(self.pos), self.r, copy.deepcopy(self.vel_vector), self.color)
        # newBall.ballID = self.ballID
        newBall.startpos = copy.deepcopy(self.startpos)
        # newBall.pos = copy.deepcopy(self.pos)
        # newBall.r = self.r
        newBall.mass = self.r
        # newBall.color = self.color
        # newBall.vel_vector = self.vel_multiplier * self.vel_vector #vel for tick
        newBall.path = copy.deepcopy(self.path)
        newBall.collisions = copy.deepcopy(self.collisions)
        newBall.predicted_collisions = copy.deepcopy(self.predicted_collisions)
        newBall.ball_time = copy.deepcopy(self.ball_time)

        return newBall
    
    def update_ball(self, t_time):
        self.pos += self.vel_vector
    
    def update_ball_pos(self, time):
        self.pos += time * self.vel_vector

    def check_wall_collisions(self):
        if self.pos[0] >= (DIM_BOARD_X - self.r) or self.pos[0] <= self.r:
            self.vel_vector[0] = -1 * self.vel_vector[0]
            self.path.append(self.pos)
            
        if self.pos[1] >= (DIM_BOARD_Y - self.r) or self.pos[1] <= self.r:
            self.vel_vector[1] = -1 * self.vel_vector[1]
            self.path.append(self.pos)

    
        

class View():
    def __init__(self):
        pass

    def render(self, screen, table):
        screen.fill((255,255,255))
        pygame.draw.line(screen, (0,255,255), (0,0), (DIM_BOARD_X,0), 5)
        pygame.draw.line(screen, (0,255,0), (0,0), (0,DIM_BOARD_Y), 5)
        pygame.draw.line(screen, (255,0,255), (0,DIM_BOARD_Y-1), (DIM_BOARD_X,DIM_BOARD_Y-1), 5)
        pygame.draw.line(screen, (255,255,0), (DIM_BOARD_X-1,0), (DIM_BOARD_X-1,DIM_BOARD_Y), 5)



        for ball in table.balls:
            # print(ball.x, ball.y)
            pygame.draw.circle(screen, ball.color, [ball.pos[0], DIM_BOARD_Y - ball.pos[1]], ball.r)

        # for ball in table.balls[0:1]:
        #     local_path = ball.path.copy()
        #     local_path.append((ball.x, ball.y))
        #     for i in range(len(local_path) - 1):
        #         pygame.draw.line(screen, (0,0,0), (local_path[i][0], DIM_BOARD_Y - local_path[i][1]), (local_path[i+1][0], DIM_BOARD_Y - local_path[i+1][1]))

        # for ball in table.balls[0:1]:
        #     for collision in ball.collisions:
        #         pygame.draw.circle(screen, (0, 0, 255), (collision[0], DIM_BOARD_Y - collision[1]), 5)

    def render_from_list(self, screen, table, t_time):
        screen.fill((255,255,255))
        pygame.draw.line(screen, (0,255,255), (0,0), (DIM_BOARD_X,0), 5)
        pygame.draw.line(screen, (0,255,0), (0,0), (0,DIM_BOARD_Y), 5)
        pygame.draw.line(screen, (255,0,255), (0,DIM_BOARD_Y-1), (DIM_BOARD_X,DIM_BOARD_Y-1), 5)
        pygame.draw.line(screen, (255,255,0), (DIM_BOARD_X-1,0), (DIM_BOARD_X-1,DIM_BOARD_Y), 5)
        # print(table.collision_list)
        # time.sleep(10)

        # for ball in table.balls:
            # print(ball.ballID, ball.vel_vector)
        # for ball in table.balls:
        #     ball.pos = copy.deepcopy(ball.startpos)
        #     ball.ball_time = 0

        # if len(table.collision_list) != 0:

        #     if t_time > table.depth:
        #         print("END!!!!")
        #         return

        #     for element in table.collision_list:
        #         # print(element)
        #         # print(t_time)
        #         # time.sleep(1)
        #         if element[1] > t_time:
        #             break
        #         else:
        #             for ball in table.balls:
        #                 if ball.ballID == element[0]:
        #                     ball.vel_vector = copy.deepcopy(element[2])
        #                     ball.pos = copy.deepcopy(element[3])
        #                     # print(ball.pos)
        #                     ball.ball_time = element[1]
        
        # for ball in table.balls:
        #     print(ball.ballID, ball.vel_vector)
        #     pygame.draw.circle(screen, ball.color, [ball.pos[0] + (ball.vel_vector[0] * (t_time - ball.ball_time)), DIM_BOARD_Y - (ball.pos[1] + (ball.vel_vector[1] * (t_time - ball.ball_time)))], ball.r)
        
        
        # pygame.display.flip()
        # time.sleep(1)
        ball_set = table.create_ball_set_to_render(t_time)

        for ball in ball_set:
            # print(ball.ballID, ball.vel_vector)
            pygame.draw.circle(screen, ball.color, [ball.pos[0], DIM_BOARD_Y - ball.pos[1]], ball.r)
        # pygame.display.flip()
        # time.sleep(1)


clock = pygame.time.Clock()

table = Table()
view = View()
# table.generate_random_ball(10)
# table.add_ball(100*multiplier, 200/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.,0.]))
# table.add_ball(424/4*multiplier, 212/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.,-0.]))
# table.add_ball(424/4*multiplier, 188/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([-0.,0.]))

# table.add_ball(100/4*multiplier, 200/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.5,0.01]))

# table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([1.,0.1]), (255,0,0))
# table.add_ball(2, np.array([500., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([-1.,0.]), (0,120,0))

table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([1.,0.1]), (255,0,0))
table.add_ball(2, np.array([500., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([-1.,0.]), (0,120,0))

# table.add_ball(500, 100 + DIM_BALL_R, DIM_BALL_R, np.array([-1.,1.]))
# table.add_ball(100, 100 + DIM_BALL_R, DIM_BALL_R, np.array([1.,1.]))

def save_image(it_num):
    filename = str(it_num)+".png"
    pygame.image.save(screen, filename)

# time.sleep(2)
# table.t_time = 
anim_time = 0
viewTable = table.copy_table()
table.predict_all_collisions(10)

whatever = 10
fps = 120
while not table.balls_stopped():
    # print(table.balls_stopped())
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False   
        if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            if fps == 120:
                fps = 60
            else:
                fps = 120


    if anim_time >= table.d_time:
        table.update_table()
        table.predict_all_collisions(1) #prediction depth

    view.render_from_list(screen, table, anim_time)
    # view.render_real_time(screen, table, table.t_time)

    # for ball in table.balls:
    #     ball.update_ball_pos(1)
    anim_time += 1
    # table.t_time+=1
    # print(table.t_time)
    # table.update_table()
    # view.render(screen, table)
    pygame.display.flip()
    pygame.display.set_caption(str(int(clock.get_fps())))
    # time.sleep(0.01)

    clock.tick(fps)

# render(screen, table)
# save_image(2)
# time.sleep(10)

