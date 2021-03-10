# import pygame
from numpy.lib.function_base import _calculate_shapes
from model.Table import Table
from model.View_pygame import View as Viewpg
from model.Ball import Ball
from model.View_tkinter import View as Viewtk
import numpy as np

# PROPERTIES 
DIM_BOARD_X = 800
DIM_BOARD_Y = 400
DIM_BALL_R = 20
anim_time = 0
fps = 60
step = 1

# INITS
table = Table(DIM_BOARD_X, DIM_BOARD_Y)
view = Viewpg(DIM_BOARD_X, DIM_BOARD_Y)   # PyGame view init
# view = Viewtk(DIM_BOARD_X, DIM_BOARD_Y)  # Tkinter view init


# BALL CONFIGURATIONS
# table.generate_random_ball(10)

# --- Two big balls
table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([1.5,0.5]), (255,0,0))
table.add_ball(2, np.array([500., 190.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([0.7,0.5]), (255,0,0))

# --- Big & small
# table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([1.,0.1]), (255,0,0))
# table.add_ball(2, np.array([500., 200.]) + DIM_BALL_R, DIM_BALL_R/4, np.array([-1.,0.]), (0,120,0))

# --- 5 balls of the same size
# table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([1.,0.1]), (255,0,0))
# table.add_ball(2, np.array([200., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([0.,0.]), (255,0,0))
# table.add_ball(3, np.array([300., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([0.,0.]), (255,0,0))
# table.add_ball(4, np.array([400., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([0.,0.]), (255,0,0))
# table.add_ball(5, np.array([500., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([-2.,0.]), (255,0,0))


# INIT PREDICTION
table.predict_all_collisions(step)


# PYGAME INIT
import pygame
pygame.init()
clock = pygame.time.Clock()


# initial_momentum = table.calculate_momentum()
# v_list = []
# for ball in table.balls:
#     v_list.append([ball])
# v_list1 = []
# v_list2 = []
# momentum_list = []
# time_list = []


breaking_var = False
while anim_time <= table.d_time and not breaking_var:
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False   
        elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            if fps == 600:
                fps = 60
            else:
                fps = 600
        elif event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            breaking_var = True

    # print(anim_time)
    table.check_for_update_prediction(anim_time, step)

    view.render_from_list(table, anim_time)
    # view.render_from_list(table, anim_time)
    # for i in range(len(table.balls)):
    #     v_list[i].append(np.linalg.norm(table.balls[i].vel_vector))
    # print(table.calculate_momentum())
    # momentum_list.append(table.calculate_momentum())
    # v_list1.append(np.linalg.norm(table.balls[0].vel_vector))
    # v_list2.append(np.linalg.norm(table.balls[1].vel_vector))
    # time_list.append(anim_time)


    anim_time += step

    pygame.display.set_caption(str(int(clock.get_fps())))

    clock.tick(fps)



# TMP ANALYSIS
# from matplotlib import pyplot as plt

# m, b = np.polyfit(time_list, momentum_list, 1)
# time_list = np.array(time_list)

# plt.subplot(311)
# plt.plot(time_list, v_list1)
# plt.subplot(312)
# plt.plot(time_list, v_list2)
# plt.subplot(313)
# plt.plot(time_list, momentum_list)
# plt.plot(time_list, [initial_momentum for _ in time_list])
# plt.plot(time_list, m*time_list + b)

# for i in range(len(v_list)):
#     plt.plot(time_list, v_list[i][1:], c = tuple([x/255 for x in v_list[i][0].color]))

# plt.show()