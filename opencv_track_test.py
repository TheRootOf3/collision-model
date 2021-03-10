import cv2
import numpy as np

import time
import pygame
import copy

from model.Table import Table
from model.View_pygame import View

def create_trackers(cap):
    tracker_list = []
    while True:
        for event in pygame.event.get():        
            if event.type == pygame.QUIT:
                running = False

        ret, frame = cap.read()
        cv2.imshow("Tracking", frame)
        output, circles = check_for_circles(frame)
        if circles is not None:
            break

    for circle in circles:
        print(circle)
        x = circle[0]
        y = circle[1]
        r = circle[2]
        
        bbox = (x-r, y-r, 2*r, 2*r)
        tracker_list.append(cv2.TrackerCSRT_create())
        tracker_list[-1].init(frame, bbox)
    return (tracker_list, circles)

def check_for_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, ksize= (9, 9), sigmaX=2)
    gray = cv2.blur(gray, (3, 3)) 
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.3, param1=100, param2=10, minDist=10, maxRadius=50)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, param1=100, param2=60, minDist=100, minRadius=30, maxRadius=50)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.5, param1=30, param2=60, minDist=20, minRadius=30, maxRadius=50)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.circle(image, (x, y), 5, (0, 128, 255), -1)
        # show the output image
        # cv2.imshow("output", np.hstack([gray]))
    return (image, circles)

def drawCircle(img,bbox):
    x, y, r = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[2]/2), int(bbox[2]/2)
    cv2.circle(img, (x,y), r, (0, 255, 0), 3)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 3 )

def bTc(bbox):
    return (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[2]/2), int(bbox[2]/2))


def register_balls(table, circles):
    id_count = 0
    for x, y, r in circles:
        table.add_ball(id_count, np.array([x, DIM_BOARD_Y - y], dtype='float64'), r, np.array([0., 0.]))
        table.balls[-1].pos_history.append(table.balls[-1].pos)
        id_count += 1


# TRACKER INITIALIZATION
cap = cv2.VideoCapture(2)
pygame.init()

tracker_list, circles = create_trackers(cap)

DIM_BOARD_X = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
DIM_BOARD_Y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(DIM_BOARD_X, DIM_BOARD_Y)
table = Table(DIM_BOARD_X, DIM_BOARD_Y)
time.sleep(1)
view = View(DIM_BOARD_X, DIM_BOARD_Y)

register_balls(table, circles)


while True:
    for event in pygame.event.get():        
        if event.type == pygame.QUIT:
            running = False  

    success, img = cap.read()
    output, circles = check_for_circles(img)
    if circles is not None:
        circle_tracked = False
        for x, y, r in circles:
            for ball in table.balls:
                if np.linalg.norm(ball.pos - np.array([x, DIM_BOARD_Y - y])) < ball.r + r + 5:
                    circle_tracked = True
            if not circle_tracked:
                print("adding a ball")
                table.add_ball(table.balls[-1].ballID + 1, np.array([x, DIM_BOARD_Y - y], dtype='float64'), r, np.array([0., 0.]))
                
                bbox = (x-r, y-r, 2*r, 2*r)
                tracker_list.append(cv2.TrackerCSRT_create())
                tracker_list[-1].init(img, bbox)
                table.balls[-1].pos_history.append(table.balls[-1].pos)



    any_success = False
    to_pop = []
    for i in range(len(tracker_list)):
        success, bbox = tracker_list[i].update(img)

        if success:
            any_success = True
            drawBox(img,bbox)
            drawCircle(img, bbox)
            x, y, r = bTc(bbox)
            if len(table.balls[i].pos_history) >= 5:
                table.balls[i].vel_vector = (table.balls[i].pos_history[-1] - table.balls[i].pos_history[-5])/5
                table.balls[i].vel_vector = (np.array([x, DIM_BOARD_Y - y], dtype='float64') - table.balls[i].pos)
            table.balls[i].pos = np.array([x, DIM_BOARD_Y - y], dtype='float64')
            table.balls[i].pos_history.append(table.balls[i].pos)
        else:
            print("removing a ball")
            to_pop.append(i)
            print(i)

    for i in to_pop[::-1]:
        tracker_list.pop(i)
        table.balls.pop(i)


          
        
    cv2.imshow("Tracking", img)
    view.render(table)
    key_pressed = cv2.waitKey(1)
    if not any_success:
        table.balls = []
        tracker_list, circles = create_trackers(cap)
        register_balls(table, circles)

    if key_pressed == ord('n'):
        table.balls = []
        tracker_list, circles = create_trackers(cap)
        register_balls(table, circles)
    
    if key_pressed == ord(' '):
        print("starting simulation!")
        # time.sleep(1)
        table.d_time = 0
        # for ball in table.balls:
        #     ball.startpos = copy.deepcopy(ball.pos)
        print(table.balls[0].pos, table.balls[0].vel_vector)
        play = True
        # for ball in table.balls:
        #     if np.linalg.norm(ball.vel_vector) == 0:
        #         play = False

        # table.predict_all_collisions(100)
        # print(table.collision_list)
        # time.sleep(1)
        out = False
        if play:
            anim_time = 0
            while table.d_time >= anim_time:
                for event in pygame.event.get():        
                    if event.type == pygame.QUIT:
                        running = False  
                    if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                        out = True

                if out:
                    break
                table.check_for_update_prediction(anim_time, 10)
                view.render_from_list(table, anim_time)
                anim_time += 1
                # time.sleep(1/30)
                time.sleep(0.05)

        table.balls = []
        tracker_list, circles = create_trackers(cap)
        register_balls(table, circles)

    if key_pressed == ord('q'):
       break

cap.release()

