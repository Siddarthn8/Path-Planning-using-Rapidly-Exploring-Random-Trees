import numpy as np
import cv2
import matplotlib.pyplot as mlp
import random
import argparse

def pixel_loc(image):
    black_pixels_loc = []
    white_pixels_loc = []
    ret, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    black_pixels = np.where(binary_img == 0)
    white_pixels = np.where(binary_img == 255)
    
    for i in range(len(black_pixels[0])):
        temp = [black_pixels[1][i], black_pixels[0][i]]
        black_pixels_loc.append(temp)

    for i in range(len(white_pixels[0])):
        temp1 = [white_pixels[1][i], white_pixels[0][i]]
        white_pixels_loc.append(temp1)
    
    return white_pixels_loc,black_pixels_loc

def randpos(white):
    ind = random.randint(0,len(white))
    x = white[ind][0]
    y = white[ind][1]
    rand = [x,y]
    return rand

def nearest(point, nodes):
    temp = []
    for n in nodes:
        temp.append(distance(n,point))
    minimum = min(temp)
    ind = temp.index(minimum)
    mnode = nodes[ind]
    return mnode

def distance(point1,point2):
    return np.sqrt(((point2[0]-point1[0])**2)+((point2[1]-point1[1])**2))

def new(near, point2, dist):
    d = distance(point2, near)
    if d>dist:
        px = near[0]+(dist*(point2[0]-near[0])/d)
        py = near[1]+(dist*(point2[1]-near[1])/d)
        pose = [int(px),int(py)]
    else:
        pose = [point2[0],point2[1]]
    return pose

def thru_obs(near,point,obs):
    temp = []
    x = [point[0], near[0]]
    y = [point[1], near[1]]    
    
    xl = min(x)
    xu = max(x)

    yl = min(y)
    yu = max(y)

    for o in obs:
        if xl<=o[0] and o[0]<=xu:
            if yl<=o[1] and o[1]<=yu:
                temp.append(o)
    
    A = point[1] - near[1]
    B = -(point[0] - near[0])
    C = -(near[0]*A)-(near[1]*B)
    den = np.sqrt(A**2 + B**2)
    cond = False

    for j in temp:
        p_dist = (A*j[0] + B*j[1] +C)/den
        if p_dist<1:
            cond = True
            break
    return cond

def RRT(image,obstacle, region, start, end, thres,clearance, iter):
    node = [start]
    temp = []
    for i in range(iter):
        # print(i)
        rand = randpos(region)
        nearnode = nearest(rand, node)
        new_node = new(nearnode, rand, thres)
        dist = min([distance(o,new_node) for o in obstacle])
        if dist > clearance:
            cond = thru_obs(nearnode, new_node, obstacle)
            cond1 = thru_obs(end, new_node, obstacle)
            if cond == False:
                node.append(new_node)
                temp.append([nearnode,new_node])
                cv2.line(image, tuple(new_node), tuple(nearnode), (0, 255, 0), 1)
                if cond1 == False:
                    node.append(end)
                    temp.append([new_node,end])
                    cv2.line(image, tuple(new_node), tuple(end), (0, 255, 0), 1)
                    break

    temp.reverse()
    path_temp = [end]
    for i in range(len(temp)):
        if path_temp[-1] in temp[i]:
            ind = temp[i].index(path_temp[-1])
            if ind == 1:
                path_temp.append(temp[i][0])
            
    path = []
    for i in range(len(path_temp)-1):
        path.append([path_temp[i],path_temp[i+1]])
    
    for i in range(len(path)):
        cv2.line(image, tuple(path[i][0]), tuple(path[i][1]),(255,0,0),2)
        
    node = np.array(node)
    node = np.reshape(node, (node.shape[0], 2))
    node = np.delete(node, 0, axis=0)
    node = np.delete(node, -1, axis=0)
    
    for i in range(node.shape[0]):
        cv2.rectangle(image, tuple(node[i]), tuple(node[i] + 4), (0, 0, 255), -1)
    cv2.circle(image, tuple(start), 4, (0, 0, 255), -1)
    cv2.circle(image, tuple(end), 4, (0, 0, 255), -1)
    image_name = args.map+'_result'
    f = 'png'
    image_path = "{}.{}".format(image_name,f)
    cv2.imwrite(image_path,image)
    cv2.imshow("RRT", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("startX", type=int)
    parser.add_argument("startY", type=int)
    parser.add_argument("goalX", type=int)
    parser.add_argument("goalY", type=int)
    args = parser.parse_args()

    img_name = args.map
    form = 'png'
    img_path = "{}.{}".format(img_name,form)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    i = cv2.imread(img_path)
    space,obs_space = pixel_loc(image=img)
    threshold = 50
    start_point = [args.startX,args.startY]
    goal_point = [args.goalX,args.goalY]
    iterations = 10000
    obs_threshold = 5
    RRT(i,obs_space,space,start_point,goal_point,threshold,obs_threshold,iterations)
