import numpy as np
import copy
from Ball import Ball
import itertools
import random
import math


class Table():
    def __init__(self, dimX, dimY):
        self.balls = []
        # self.friction = 0.93  # 1 = no friction 
        # self.friction = 0.9999
        self.friction = 1
        # self.t_time = 0
        self.collision_list = []
        self.d_time = 0
        self.dimX = dimX
        self.dimY = dimY


    def calculate_momentum(self):
        ''' momentum of the system '''
        momentum = 0
        for ball in self.balls:
            momentum += np.linalg.norm(ball.vel_vector) * ball.mass
        #     print(np.linalg.norm(ball.vel_vector), ball.mass)
        # print("-------------")
        return momentum


    def copy_table(self):
        ''' creates clone of the current table obj'''
        newTable = Table(self.dimX, self.dimY)
        newTable.balls = copy.deepcopy(self.balls)
        newTable.friction = self.friction
        # newTable.t_time = self.t_time
        newTable.d_time = self.d_time
        
        return newTable


    def add_ball(self, ballID, pos, r, vel_vector = np.array([1., 1.]), color = (255, 0, 0)):
        ''' add ball to the table obj'''
        self.balls.append(Ball(ballID, pos, r, vel_vector, color))
    

    def check_for_update_prediction(self, anim_time, lookahead):
        ''' check if table requires further predictions '''
        if anim_time >= self.d_time:
            self.update_table()
            self.predict_all_collisions(lookahead) #prediction depth


    def update_table(self):
        ''' update table with current state '''
        # print("abc")
        if len(self.collision_list) != 0:
            for element in self.collision_list:
                    for ball in self.balls:
                        if ball.ballID == element[0]:
                            ball.vel_vector = copy.deepcopy(element[2].vel_vector)
                            ball.pos = copy.deepcopy(element[2].pos)
                            # print(ball.pos)
                            ball.ball_time = element[1]
                
        for ball in self.balls:
            # print(ball.pos, ball.vel_vector)
            ball.update_ball_pos(self.d_time - ball.ball_time)
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
        ''' predict collisions for every ball in given depth time '''
        tmp_depth_time = depth_time

        self.collision_list = []
        pred_table = self.copy_table() # Create copy of the table for predictions

        while depth_time >= 0:
            ''' find all collisions until depth time runs out '''
            
            first_col = pred_table.predict_wall_ball() # Find the first collision between any two objects

            if first_col[-1] > depth_time: # If the collision occurs after the remaining depth time, break
                break

            for ball in pred_table.balls:
                ''' update all balls on the table copy '''
                ball.update_ball_pos(first_col[-1]) 

            if first_col[0] == 0:
                ''' if the collision occurs between a ball and a wall '''
                # for ball in pred_table.balls:
                #     ball.update_ball_pos(first_col[-1])
                first_col[1].vel_vector = pred_table.return_new_wall_vector(first_col[1], first_col[2])

                depth_time -= first_col[-1]
                newball = first_col[1].copy_ball()
            
                self.collision_list.append([first_col[1].ballID, pred_table.d_time + first_col[-1], newball])
                pred_table.d_time += first_col[-1]

            else:
                ''' if the collision occurs between two balls '''
                # print(np.linalg.norm(first_col[1].vel_vector), np.linalg.norm(first_col[2].vel_vector))
                first_col[1].vel_vector, first_col[2].vel_vector = pred_table.calculate_new_vectors(first_col[1].mass, first_col[2].mass, first_col[1].vel_vector, first_col[2].vel_vector, first_col[1].pos, first_col[2].pos)
                # print(np.linalg.norm(first_col[1].vel_vector), np.linalg.norm(first_col[2].vel_vector))
                # print(np.linalg.norm(first_col[1].pos - first_col[2].pos))
                # print("###")
                depth_time -= first_col[-1]
                newball1 = first_col[1].copy_ball()
                newball2 = first_col[2].copy_ball()

                self.collision_list.append([first_col[1].ballID, pred_table.d_time + first_col[-1], newball1])
                self.collision_list.append([first_col[2].ballID, pred_table.d_time + first_col[-1], newball2])
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
        ''' new velocity vectors after collision between a ball and a wall '''

        if wall < 2:
            return (ball.vel_vector * np.array([-1,1]))
        else:
            return (ball.vel_vector * np.array([1,-1]))

    def calculate_ball_col_time(self, ball1, ball2):
        ''' calculating in what time there will be a collision between two balls'''
        # print(np.linalg.norm(ball1.pos - ball2.pos))
        p = [np.dot((ball1.vel_vector - ball2.vel_vector), (ball1.vel_vector - ball2.vel_vector)),
            2*np.dot(ball1.vel_vector - ball2.vel_vector, ball1.pos - ball2.pos),
            np.linalg.norm(ball1.pos - ball2.pos)**2 - (ball1.r + ball2.r)**2
        ]


        roots = np.roots(p)
        correct_roots = []
        for x in roots:
            # print(np.linalg.norm((ball1.pos + x*ball1.vel_vector) - (ball2.pos + x*ball2.vel_vector)), ball1.r + ball2.r, (np.linalg.norm((ball1.pos + x*ball1.vel_vector) - (ball2.pos + x*ball2.vel_vector)) >= ball1.r + ball2.r))
            if self.check_calculated_time(x, ball1) and self.check_calculated_time(x, ball2):
                # print("yes")
                prec = 13
                # print(np.linalg.norm((ball1.pos + x*ball1.vel_vector) - (ball2.pos + x*ball2.vel_vector)), ball1.r + ball2.r)
                while np.linalg.norm((ball1.pos + (x * ball1.vel_vector)) - (ball2.pos + (x * ball2.vel_vector))) < ball1.r + ball2.r and prec > 0:
                    x = self.round_time_down(x, prec)
                    prec -= 1
                    # print(np.linalg.norm((ball1.pos + x*ball1.vel_vector) - (ball2.pos + x*ball2.vel_vector)), ball1.r + ball2.r)
                    # print("LOOOOL")
                correct_roots.append(x)

        if len(correct_roots) == 0:
            return None
        else:
            return min(correct_roots)

    def round_time_down(self, time, precision):
        return math.floor(time * (10 ** precision)) / (10 ** precision)


    def check_calculated_time(self, time, ball):
        # print(time)
        # print(ball.pos, ball.vel_vector)
        if time <= 0 or isinstance(time, complex): #added time <= 0.00001 to get rid of two balls blocking. May cause problems!
            # print("NOPE1")
            return False
        if ball.pos[0] + time*ball.vel_vector[0] >= 0 + ball.r and ball.pos[0] + time*ball.vel_vector[0] <= self.dimX - ball.r and\
            ball.pos[1] + time*ball.vel_vector[1] >= 0 + ball.r and ball.pos[1] + time*ball.vel_vector[1] <= self.dimY - ball.r:
            # print("YUP")
            return True
        # print("NOPE2")
        return False


    def calculate_wall_col_time(self, ball):
        times = [1000000, 1000000, 1000000, 1000000]
        if ball.vel_vector[0] != 0:
            times[0] = (0 + ball.r - ball.pos[0]) / ball.vel_vector[0]
            times[1] = (self.dimX - ball.r - ball.pos[0]) / ball.vel_vector[0]
        
        if ball.vel_vector[1] != 0:
            times[2] = (0 + ball.r - ball.pos[1]) / ball.vel_vector[1]
            times[3] = (self.dimY - ball.r - ball.pos[1]) / ball.vel_vector[1]

        col_time = 1000000
        wall = None # wall 0 - left, wall 1 - right, wall 2 - bottom, wall 3 - top
        for i in range(len(times)):
            if times[i] < col_time and times[i] > 9e-13: #added time <= 0.00001 to get rid of two balls blocking. May cause problems!
                col_time = times[i]
                wall = i
        
        # print(times)
        # print(self.t_time, ball.ballID, col_time, wall)
        
        return (col_time, wall)

        
    def generate_cords(self):
        r = random.randint(20, 60)
        pos = np.array([random.uniform(r, self.dimX - r), random.uniform(r, self.dimY - r)])

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
                np.array([random.uniform(-4, 4), random.uniform(-4, 4)]),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                ))            

    def calculate_new_vectors(self, m1, m2, v1, v2, pos1, pos2):
        ''' new velocity vectors after collision between two balls '''
        v_new1 = v1 - 2 * m2/(m1 + m2) * (np.dot(v1 - v2, pos1 - pos2)/((np.linalg.norm(pos1 - pos2))) ** 2)*(pos1 - pos2)
        v_new2 = v2 - 2 * m1/(m1 + m2) * (np.dot(v2 - v1, pos2 - pos1)/((np.linalg.norm(pos2 - pos1))) ** 2)*(pos2 - pos1)
        return (v_new1, v_new2)

    def create_ball_set_to_render(self, t_time2):
        ''' create ball objects with appropriate positions to draw ''' 
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
                            ball.vel_vector = copy.deepcopy(element[2].vel_vector)
                            ball.pos = copy.deepcopy(element[2].pos)
                            # print(ball.pos)
                            ball.ball_time = element[1]
        
        for ball in ball_set:
            ball.pos[0] = ball.pos[0] + (ball.vel_vector[0] * (t_time2 - ball.ball_time))
            ball.pos[1] = ball.pos[1] + (ball.vel_vector[1] * (t_time2 - ball.ball_time))
        # for ball in self.balls:
            # print(ball.ballID, ball.pos)
        # time.sleep(10)
        # print(ball_set)
        return ball_set