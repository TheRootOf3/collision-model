import tkinter as tk


class View:
    def __init__(self, dimX, dimY):
        self.dimX = dimX
        self.dimY = dimY

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root,
                                width=self.dimX,
                                height=self.dimY)
        self.canvas.pack()

        self.canvas.create_line(0, 0, self.dimX, 0, fill='#00FFFF')
        self.canvas.create_line(0, 0, 0, self.dimY, fill='#00FF00')
        self.canvas.create_line(0, self.dimY - 1, self.dimX, self.dimY - 1, fill='#FF00FF')
        self.canvas.create_line(self.dimX - 1, 0, self.dimX - 1, self.dimY, fill='#FFFF00')
        self.root.update()

    def render_from_list(self, table, t_time):
        self.canvas.delete("all")
        self.canvas.create_line(0, 0, self.dimX, 0, fill='#00FFFF')
        self.canvas.create_line(0, 0, 0, self.dimY, fill='#00FF00')
        self.canvas.create_line(0, self.dimY - 1, self.dimX, self.dimY - 1, fill='#FF00FF')
        self.canvas.create_line(self.dimX - 1, 0, self.dimX - 1, self.dimY, fill='#FFFF00')

        ball_set = table.create_ball_set_to_render(t_time)

        for ball in ball_set:
            self.canvas.create_oval(ball.pos[0] - ball.r, self.dimY - ball.pos[1] - ball.r, ball.pos[0] + ball.r,
                                    self.dimY - ball.pos[1] + ball.r, fill=('#%02x%02x%02x' % ball.color))
        self.root.update()
