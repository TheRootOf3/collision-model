import copy


class Ball:
    def __init__(self, ballID, pos, r, vel_vector, color):
        self.ballID = ballID
        self.startpos = pos
        self.pos = pos
        self.r = r
        self.mass = r ** 3
        self.color = color
        self.vel_vector = vel_vector  # vel for tick
        self.path = [pos]
        self.collisions = []
        self.predicted_collisions = []
        self.ball_time = 0
        self.pos_history = []

    def copy_ball(self):
        newBall = Ball(self.ballID, copy.deepcopy(self.pos), self.r, copy.deepcopy(self.vel_vector), self.color)
        # newBall.ballID = self.ballID
        newBall.startpos = copy.deepcopy(self.startpos)
        # newBall.pos = copy.deepcopy(self.pos)
        # newBall.r = self.r
        newBall.mass = self.mass
        # newBall.color = self.color
        # newBall.vel_vector = self.vel_multiplier * self.vel_vector #vel for tick
        newBall.path = copy.deepcopy(self.path)
        newBall.collisions = copy.deepcopy(self.collisions)
        newBall.predicted_collisions = copy.deepcopy(self.predicted_collisions)
        newBall.ball_time = copy.deepcopy(self.ball_time)

        return newBall

    def update_ball(self):
        self.pos += self.vel_vector

    def update_ball_pos(self, time):
        self.pos += time * self.vel_vector
