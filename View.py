import pygame

class View():
    def __init__(self):
        pass

    def render(self, screen, table):
        dimX = table.dimX
        dimY = table.dimY
        screen.fill((255,255,255))
        pygame.draw.line(screen, (0,255,255), (0,0), (dimX,0), 5)
        pygame.draw.line(screen, (0,255,0), (0,0), (0,dimY), 5)
        pygame.draw.line(screen, (255,0,255), (0,dimY-1), (dimX,dimY-1), 5)
        pygame.draw.line(screen, (255,255,0), (dimX-1,0), (dimX-1,dimY), 5)

        for ball in table.balls:
            # print(ball.x, ball.y)
            pygame.draw.circle(screen, ball.color, [ball.pos[0], dimY - ball.pos[1]], ball.r)
        pygame.display.flip()

        # for ball in table.balls[0:1]:
        #     local_path = ball.path.copy()
        #     local_path.append((ball.x, ball.y))
        #     for i in range(len(local_path) - 1):
        #         pygame.draw.line(screen, (0,0,0), (local_path[i][0], DIM_BOARD_Y - local_path[i][1]), (local_path[i+1][0], DIM_BOARD_Y - local_path[i+1][1]))

        # for ball in table.balls[0:1]:
        #     for collision in ball.collisions:
        #         pygame.draw.circle(screen, (0, 0, 255), (collision[0], DIM_BOARD_Y - collision[1]), 5)

    def render_from_list(self, screen, table, t_time):
        dimX = table.dimX
        dimY = table.dimY
        screen.fill((255,255,255))
        pygame.draw.line(screen, (0,255,255), (0,0), (dimX,0), 5)
        pygame.draw.line(screen, (0,255,0), (0,0), (0,dimY), 5)
        pygame.draw.line(screen, (255,0,255), (0,dimY-1), (dimX,dimY-1), 5)
        pygame.draw.line(screen, (255,255,0), (dimX-1,0), (dimX-1,dimY), 5)
        
        ball_set = table.create_ball_set_to_render(t_time)

        for ball in ball_set:
            pygame.draw.circle(screen, ball.color, [ball.pos[0], dimY - ball.pos[1]], ball.r)
        pygame.display.flip()
            