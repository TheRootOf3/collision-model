# import pygame
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
fps = 60
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


while anim_time <= table.d_time:
    # print(table.balls_stopped())
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False   
        if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            if fps == 600:
                fps = 60
            else:
                fps = 600

    # print(anim_time)
    table.check_for_update_prediction(anim_time, step)
    # if anim_time >= table.d_time:
    #     table.update_table()
    #     table.predict_all_collisions(1) #prediction depth

    view.render_from_list(screen, table, anim_time)
    print(table.calculate_momentum())
    # view.render_real_time(screen, table, table.t_time)

    # for ball in table.balls:
    #     ball.update_ball_pos(1)
    print(anim_time)
    anim_time += step
    # table.t_time+=1
    # print(table.t_time)
    # table.update_table()
    # view.render(screen, table)
    # pygame.display.flip()
    pygame.display.set_caption(str(int(clock.get_fps())))
    # time.sleep(0.01)

    clock.tick(fps)

# render(screen, table)
# save_image(2)
# time.sleep(10)

