# import pygame
from numpy.lib.function_base import _calculate_shapes
from Table import Table
from View import View
from Ball import Ball
import numpy as np

# multiplier = 4

# DIM_BOARD_X = 224*multiplier
# DIM_BOARD_Y = 112*multiplier
# DIM_BALL_R = math.ceil(2.8*multiplier)


# PROPERTIES 
DIM_BOARD_X = 800
DIM_BOARD_Y = 400
DIM_BALL_R = 20

anim_time = 0
fps = 600
step = 1


table = Table(DIM_BOARD_X, DIM_BOARD_Y)
view = View()
table.generate_random_ball(10)
# table.add_ball(100*multiplier, 200/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.,0.]))
# table.add_ball(424/4*multiplier, 212/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.,-0.]))
# table.add_ball(424/4*multiplier, 188/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([-0.,0.]))

# table.add_ball(100/4*multiplier, 200/4*multiplier + DIM_BALL_R, DIM_BALL_R, np.array([0.5,0.01]))

# table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([1.,0.1]), (255,0,0))
# table.add_ball(2, np.array([500., 200.]) + DIM_BALL_R, DIM_BALL_R, np.array([-1.,0.]), (0,120,0))

# table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([1.5,0.5]), (255,0,0))
# table.add_ball(2, np.array([500., 190.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([0.7,0.5]), (255,0,0))

# table.add_ball(1, np.array([100., 200.]) + DIM_BALL_R, DIM_BALL_R*6, np.array([1.,0.1]), (255,0,0))
# table.add_ball(2, np.array([500., 200.]) + DIM_BALL_R, DIM_BALL_R/4, np.array([-1.,0.]), (0,120,0))

# table.add_ball(500, 100 + DIM_BALL_R, DIM_BALL_R, np.array([-1.,1.]))
# table.add_ball(100, 100 + DIM_BALL_R, DIM_BALL_R, np.array([1.,1.]))

# def save_image(it_num):
#     filename = str(it_num)+".png"
#     pygame.image.save(screen, filename)



# Initial predicition
table.predict_all_collisions(step)


# PyGame Init for visualization
import pygame
pygame.init()
screen = pygame.display.set_mode([DIM_BOARD_X, DIM_BOARD_Y])
screen.fill((255,255,255))

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

    view.render_from_list(screen, table, anim_time)
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