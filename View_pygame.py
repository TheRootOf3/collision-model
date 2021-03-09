import pygame

class View():
    def __init__(self, dimX, dimY):
        self.dimX = dimX
        self.dimY = dimY
        self.screen = pygame.display.set_mode([dimX, dimY])
        self.screen.fill((255,255,255))
        pygame.display.flip()

    def render(self, table):
        self.screen.fill((255,255,255))
        pygame.draw.line(self.screen, (0,255,255), (0,0), (self.dimX,0), 5)
        pygame.draw.line(self.screen, (0,255,0), (0,0), (0,self.dimY), 5)
        pygame.draw.line(self.screen, (255,0,255), (0,self.dimY-1), (self.dimX,self.dimY-1), 5)
        pygame.draw.line(self.screen, (255,255,0), (self.dimX-1,0), (self.dimX-1,self.dimY), 5)

        for ball in table.balls:
            # print(ball.x, ball.y)
            pygame.draw.circle(self.screen, ball.color, [ball.pos[0], self.dimY - ball.pos[1]], ball.r)
        pygame.display.flip()

    def render_from_list(self, table, t_time):
        self.screen.fill((255,255,255))
        pygame.draw.line(self.screen, (0,255,255), (0,0), (self.dimX,0), 5)
        pygame.draw.line(self.screen, (0,255,0), (0,0), (0,self.dimY), 5)
        pygame.draw.line(self.screen, (255,0,255), (0,self.dimY-1), (self.dimX,self.dimY-1), 5)
        pygame.draw.line(self.screen, (255,255,0), (self.dimX-1,0), (self.dimX-1,self.dimY), 5)
        
        ball_set = table.create_ball_set_to_render(t_time)

        for ball in ball_set:
            pygame.draw.circle(self.screen, ball.color, [ball.pos[0], self.dimY - ball.pos[1]], ball.r)
        pygame.display.flip()
            