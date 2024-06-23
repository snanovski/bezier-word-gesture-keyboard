import numpy as np
import bezier
from scipy.optimize import minimize, basinhopping
import scipy
import matplotlib.pyplot as plt
import time
import random
import math

word_endpoint_dictionary = {}

def load_word_endpoints_dictionary():
    file = open("./Data/graph_qwerty_direct.txt")
    #file = open("graph_qwerty_line_20.txt")
    #file = open("graph_qwerty_beziercubic_20.txt")
    for line in file.readlines():
        parts = line.split(":")
        coords = list(map(float, parts[1].split(",")))
        #coords_normalized = list(map(float, parts[2].split(",")))
        #word_endpoint_dictionary[parts[0]] = (np.vstack((coords[0::2], coords[1::2])).T, np.vstack((coords_normalized[0::2], coords_normalized[1::2])).T)
        word_endpoint_dictionary[parts[0]] = np.vstack((coords[0::2], coords[1::2])).T

training_data = []

def load_training_data(filename):
    file = open(filename)
    for i, line in enumerate(file.readlines()):
        #if (i == 500):
        #    break
        if (line.startswith("=")):
            continue        
        parts = line.split(":")
        if (len(parts[0]) == 1):
            continue
        coords = list(map(float, parts[1].split(",")))
        steppedCoords = getUserInputStepPoints(np.vstack((coords[0::2], coords[1::2])).T, 20)
        training_data.append((parts[0], steppedCoords))

def normalizePoints(points):
    bounding_box = [min(points[:,0]), max(points[:,0]), min(points[:,1]), max(points[:,1])]
    bounding_box_size = [bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]]

    if max(bounding_box_size[0], bounding_box_size[1]) != 0:
        s = 2 / max(bounding_box_size[0], bounding_box_size[1])
    else:
        s = 1

    middle_point = np.array([(bounding_box[0] + bounding_box[1]) / 2, (bounding_box[2] + bounding_box[3]) / 2])

    return (points - middle_point)*s  

def getUserInputStepPoints(points, steps):
    lengths = []
    length = 0
    for i in range(len(points) - 1):
        v = points[i+1] - points[i]
        l = math.sqrt(v[0]*v[0] + v[1]*v[1])
        lengths.append(l)
        length += l

    if (length == 0):
        return np.fill((steps,), points[0])

    step_size = length / (steps-1)
    cur_step = 0
    cur_line = 0
    ret = np.zeros((steps, 2))
    ret[0] = points[0]
    for cur_point in range(1, steps):
        cur_step += step_size
        while (cur_step > lengths[cur_line] and cur_line < len(lengths) - 1):
            cur_step -= lengths[cur_line]
            cur_line += 1
        ret[cur_point] = points[cur_line] * (1 - (cur_step / lengths[cur_line])) + points[cur_line + 1] * (cur_step / lengths[cur_line])

    return ret, normalizePoints(ret)

def shapeCost(normalizedInput, normalizedTemplate):
    return np.sum(np.sqrt(np.sum(np.power((normalizedInput - normalizedTemplate), 2), axis=1))) / normalizedInput.shape[0]

def locationCost(input, template):
    steps = input.shape[0]; 
    cost = 0
    d = 0
    d2 = 0
    for pt in template:
        minn = float("inf")
        for pi in input:
            v = pt - pi
            m = math.sqrt(v[0]*v[0] + v[1]*v[1])
            minn = min(minn, m)
        d += max(minn - keyRadius, 0)

    for pi in template:
        minn = float("inf")
        for pt in input:
            v = pt - pi
            m = math.sqrt(v[0]*v[0] + v[1]*v[1])
            minn = min(minn, m)
        d2 += max(minn - keyRadius, 0)

    if (d == 0 and d2 == 0): # this part can be improved significantly
        cost = 0
    else:
        alpha = np.zeros((steps))
        for i in range(math.ceil(steps / 2)):
            a = abs(0.5 - (1.0 / steps) * i)
            alpha[i] = a + 10
            alpha[steps - 1 - i] = a + 10    
        alpha /= np.sum(alpha)

        for i in range(input.shape[0]):
            v = input[i] - template[i]
            cost += math.sqrt(v[0]*v[0] + v[1]*v[1]) * alpha[i]

    return cost
    
def channelIntegration(shapeCost, locationCost):
    shapeProb = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-shapeCost*shapeCost / sigma / sigma / 2)
    locationProb = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-locationCost*locationCost / sigma / sigma / 2)
    return shapeProb * locationProb

def get_all_bezier_control_points(first_point, last_point, parameters):
    p0x, p0y = first_point[0], first_point[1]
    pnx, pny = last_point[0], last_point[1]
    length = 2 + parameters.shape[0]
    dx, dy = pnx - p0x, pny - p0y
    full_curve = np.zeros((length, 2))
    full_curve[0,] = [p0x, p0y]
    full_curve[length-1,] = [pnx, pny]
    for i in range(0, parameters.shape[0]):
        pix = parameters[i][0]
        piy = parameters[i][1]
        full_curve[i+1,] = [p0x + pix * dx - piy * dy, p0y + pix * dy + piy * dx]                
    return full_curve

def getSteppedBezierCurve(word, params):
    length = 0
    curves = []
    curve_lengths = []
    for i in range(word.shape[0] - 1):
        curve_points = get_all_bezier_control_points(word[i], word[i+1], params) 
        #print(curve_points)
        nodes = np.asfortranarray(curve_points.T)
        curve = bezier.Curve(nodes, degree = curve_points.shape[0] - 1)                           
        length += curve.length
        curve_lengths.append(curve.length)
        curves.append(curve)
        
    step_size = length / (steps - 1)
    cur_curve = 0
    cur_pos = 0
    step_points = np.zeros((steps, 2))
    step_points[0] = word[0]
    for i in range(1, steps):
        cur_pos += step_size            
        while (cur_pos > curve_lengths[cur_curve] + 0.00001):
            cur_pos -= curve_lengths[cur_curve]
            cur_curve += 1
        point = curves[cur_curve].evaluate(cur_pos/curve_lengths[cur_curve])[:,0]
        step_points[i] = point

    return step_points, normalizePoints(step_points)

def getScoreForSample(sample, params):
    (word, (regular, normalized)) = sample
    (regularTemplate, normalizedTemplate) = getSteppedBezierCurve(word_endpoint_dictionary[word], params)    
    
    sc = shapeCost(normalized, normalizedTemplate)
    lc = locationCost(regular, regularTemplate)
    return channelIntegration(sc, lc)    

def getAverageScore(params):
    ans = 0
    for sample in training_data:
        ans += getScoreForSample(sample, params)
    return ans / len(training_data)

sigma = keyRadius = 1/10
steps = 20

load_word_endpoints_dictionary()
#load_training_data("user_data.txt")
load_training_data("./Data/user_training_data_first_half.txt")

#print(getAverageScore(np.array([[0.29130457, -0.01436536], [0.78482133,  0.04207516]])))

current_best = float("-inf")


def opt_fun(p): 
    global current_best
    params = np.vstack((p[0::2], p[1::2])).T
    score = getAverageScore(params)
    if (score > current_best):
        current_best = score
        print(f"New best score of {score} found with parameters = {p}")
    return -score

# getSteppedBezierCurve(word_endpoint_dictionary["hello"], np.array([[1/3, 1/3], [2/3, -1/3]]))

optimal_params = np.array([[0.291, -0.014], [0.785, 0.042]])
norm_factor = (1/sigma/math.sqrt(2*math.pi))**2
print(norm_factor)
scores_opt = []
scores_lines = []
for sample in training_data:
    s = getScoreForSample(sample, optimal_params) / norm_factor
    scores_opt.append(s)
    s = getScoreForSample(sample, np.array([])) / norm_factor
    scores_lines.append(s)

print(f"Bezier: Mean = {np.mean(scores_opt)}, STD = {np.std(scores_opt)}, Count = {len(scores_opt)}")
print(f"Lines: Mean = {np.mean(scores_lines)}, STD = {np.std(scores_lines)}, Count = {len(scores_lines)}")

print(-getAverageScore(np.array([])))

start = time.time()

minimised = basinhopping(opt_fun, (1/3, 0, 2/3, 0))
end = time.time()

print("RESULT:")
print(minimised)
print(f"Time = {end - start}")

#print(getAverageScore(np.array([[0.3, 0.2], [0.7, -0.2]])))