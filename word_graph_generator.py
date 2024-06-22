import argparse
import math
import os
import time

import numpy as np
import scipy

from bezier_curve_finder import get_all_bezier_control_points
import bezier

class WordGraphGenerator:
    def __init__(self, layout, layouts_file_path):
        self.layout = layout
        self.keyboardLength = 1
        self.layouts = []
        self.layoutKeys = {}
        self.availableChars = []
        self.read_layouts_file(layouts_file_path)
        if layout not in self.layouts:
            print("Layout (", layout, ") not found. Maybe you wrote the layout's name wrong.")
            exit()

        for i in range(0, len(self.layoutKeys[self.layout][0])):
            line_length = len(self.layoutKeys[self.layout][1][i]) + np.abs(self.layoutKeys[self.layout][0][i])
            if " " in self.layoutKeys[self.layout][1][i]:
                line_length += 7  # because length of spacebar is 8 * normal keysize, that means 7 * keysize extra
            if "<" in self.layoutKeys[self.layout][1][i]:
                line_length += 1  # because length of backspace is 2 * normal keysize, that means 1 * keysize extra
            if line_length > self.keyboardLength:
                self.keyboardLength = line_length

    # returns a dictionary with all the characters of the layout as keys and its positions as value.
    def get_character_positions(self):
        letter_pos = {}
        for y in range(0, len(self.layoutKeys[self.layout][0])):
            x = self.layoutKeys[self.layout][0][len(self.layoutKeys[self.layout][0]) - 1 - y]

            for letter in self.layoutKeys[self.layout][1][len(self.layoutKeys[self.layout][0]) - 1 - y]:
                if letter == " ":
                    x += 8
                    continue
                if letter == "<":
                    x += 2
                    continue
                letter_pos[letter.lower()] = np.array([x, y])
                x += 1
        return letter_pos

    # returns a list of points for the given word (points where its letters lie).
    # "word" is the string for which the character positions should be returned
    # "letterPos" is the dictionary containing all the characters and their positions
    @staticmethod
    def get_points_for_word(word, letter_pos):
        points = []
        for letter in word.lower():
            points.append(letter_pos.get(letter))
        return points

    # returns summed up length for all distances between all given points.
    # "pointsArr" is an array of 2D points
    @staticmethod
    def get_length_by_points(points_arr):
        dist = 0
        for i in range(0, len(points_arr) - 1):
            dist_vec = points_arr[i] - points_arr[i + 1]
            dist += math.sqrt(dist_vec[0] ** 2 + dist_vec[1] ** 2)
        return dist
    
    # gets the distances from the start to each point, from 0 to 1
    @staticmethod
    def get_length_from_start_normalised(points_arr):
        dist = 0
        ret = [dist]
        total_length = WordGraphGenerator.get_length_by_points(points_arr)
        for i in range(0, len(points_arr) - 1):
            dist_vec = points_arr[i] - points_arr[i + 1]
            dist += math.sqrt(dist_vec[0] ** 2 + dist_vec[1] ** 2) / total_length
            ret.append(dist)
        return ret


    # returns the "steps" sampled points for the given string (word).
    # "word" is a string
    # "steps" is the number of sampling points
    # "letterPos" is the dictionary containing all the characters and their positions
    def get_word_graph_step_point(self, word, steps, letter_pos):

        # checks that the word can be spelled out with the available characters
        for letter in word:
            if letter not in self.availableChars:
                return None, None
        letter_points = self.get_points_for_word(word, letter_pos)
        length = self.get_length_by_points(letter_points)
        if length == 0:
            step_points = []
            curr_pos = letter_points[0]
            for i in range(0, steps):
                step_points.append((curr_pos + np.array([0.5, 0.5])) / self.keyboardLength)
            step_points_normalized = self.normalize(step_points, 1)
            return step_points, step_points_normalized

        step_size = length / (steps - 1)
        dist_vecs = []
        for i in range(0, len(letter_points) - 1):
            dist_vecs.append(letter_points[i + 1] - letter_points[i])

        num_steps = 1
        curr_step = step_size
        curr_pos = letter_points[0]
        curr_pos_num = 0
        curr_dist_vec_num = 0

        step_points = [(curr_pos + np.array([0.5, 0.5])) / self.keyboardLength]

        while num_steps < steps:
            dist_vec = dist_vecs[curr_dist_vec_num]
            dist_vec_length = math.sqrt(dist_vec[0] ** 2 + dist_vec[1] ** 2)  # much faster than using np.linalg.norm()
            if curr_step != step_size:
                if dist_vec_length - curr_step > -0.00001:  # error for abandoned and acknowledged was here
                    num_steps += 1
                    curr_pos = curr_pos + dist_vec / dist_vec_length * curr_step
                    # calculate new distance vector
                    dist_vecs[curr_dist_vec_num] = letter_points[curr_pos_num + 1] - curr_pos
                    step_points.append((curr_pos + np.array([0.5, 0.5])) / self.keyboardLength)
                    curr_step = step_size
                else:
                    curr_step -= dist_vec_length
                    curr_dist_vec_num += 1
                    curr_pos_num += 1
                    curr_pos = letter_points[curr_pos_num]

            elif int(dist_vec_length / step_size + 0.00001) > 0:  # adding 0.00001 to avoid rounding errors
                num_points_on_line = int(dist_vec_length / step_size + 0.00001)
                num_steps += num_points_on_line
                for i in range(0, num_points_on_line):
                    step_points.append(((curr_pos + (i + 1) * (dist_vec / dist_vec_length * step_size)) + np.array(
                        [0.5, 0.5])) / self.keyboardLength)

                if dist_vec_length - num_points_on_line * step_size > 0.00001:
                    curr_step -= (dist_vec_length - num_points_on_line * step_size)

                curr_dist_vec_num += 1
                curr_pos_num += 1
                curr_pos = letter_points[curr_pos_num]

            else:
                curr_step -= dist_vec_length
                curr_dist_vec_num += 1
                curr_pos_num += 1
                curr_pos = letter_points[curr_pos_num]

        step_points_normalized = self.normalize(step_points, 2)
        return step_points, step_points_normalized
    



    # returns the "steps" sampled points for the given string (word).
    # "word" is a string
    # "steps" is the number of sampling points
    # "letterPos" is the dictionary containing all the characters and their positions
    def get_word_graph_step_point_splines(self, word, steps, letter_pos):
        
        # checks that the word can be spelled out with the available characters
        no_duplicates_word = ""
        for letter in word:
            if letter not in self.availableChars:
                return None, None
            if (no_duplicates_word == "" or no_duplicates_word[len(no_duplicates_word) - 1] != letter):
                no_duplicates_word += letter

        word = no_duplicates_word

        if (len(word) <= 3):
            return self.get_word_graph_step_point(word, steps, letter_pos)

        letter_points = self.get_points_for_word(word, letter_pos)
        length = self.get_length_by_points(letter_points)
        if length == 0:
            step_points = []
            curr_pos = letter_points[0]
            for i in range(0, steps):
                step_points.append((curr_pos + np.array([0.5, 0.5])) / self.keyboardLength)
            step_points_normalized = self.normalize(step_points, 1)
            return step_points, step_points_normalized

        t = WordGraphGenerator.get_length_from_start_normalised(letter_points)
        letter_points = np.array(letter_points)
        x, y = letter_points[:,0], letter_points[:,1]
        #print(x)
        #print(y)
        #print(t)
        spl = scipy.interpolate.make_interp_spline(t, np.c_[x, y])

        t_new = np.linspace(0, 1, steps)
        x_new, y_new = spl(t_new).T

        #print(x_new)
        #print(y_new)
        step_points = (np.c_[x_new, y_new] + 0.5) / self.keyboardLength
        #print(step_points)
        step_points_normalized = self.normalize(step_points, 2)

        return step_points, step_points_normalized
    
    def get_word_graph_step_bezier_curve(self, word, steps, letter_pos, bezier_params):

        # checks that the word can be spelled out with the available characters
        for letter in word:
            if letter not in self.availableChars:
                return None, None
        letter_points = self.get_points_for_word(word, letter_pos)
        self.get_length_by_points(letter_points)
        if self.get_length_by_points(letter_points) == 0:
            step_points = []
            curr_pos = letter_points[0]
            for i in range(0, steps):
                step_points.append((curr_pos + np.array([0.5, 0.5])) / self.keyboardLength)
            step_points_normalized = self.normalize(step_points, 1)
            return step_points, step_points_normalized


        length = 0
        curves = []
        curve_lengths = []
        for i in range(len(letter_points) - 1):
            curve_points = get_all_bezier_control_points([letter_points[i][0], letter_points[i][1]], [letter_points[i+1][0], letter_points[i+1][1]], bezier_params) 
            nodes = np.asfortranarray([
                curve_points[0::2],
                curve_points[1::2],
            ])
            curve = bezier.Curve(nodes, degree = len(curve_points)/2 - 1)                           
            length += curve.length
            curve_lengths.append(curve.length)
            curves.append(curve)

        #length = self.get_length_by_points(letter_points)

        step_size = length / (steps - 1)
        cur_curve = 0
        cur_pos = 0
        step_points = [(letter_points[0] + np.array([0.5, 0.5])) / self.keyboardLength]

        for _ in range(steps - 1):
            cur_pos += step_size            

            while (cur_pos > curve_lengths[cur_curve] + 0.00001):
                cur_pos -= curve_lengths[cur_curve]
                cur_curve += 1

            point = curves[cur_curve].evaluate(cur_pos/curve_lengths[cur_curve])[:,0]
            #print(point)
            step_points.append((point + np.array([0.5, 0.5])) / self.keyboardLength)

        # num_steps = 1
        # curr_step = step_size
        # curr_pos = letter_points[0]
        # curr_pos_num = 0
        # curr_dist_vec_num = 0

        # step_points = [(curr_pos + np.array([0.5, 0.5])) / self.keyboardLength]

        # while num_steps < steps:
        #     dist_vec_length = dist_vecs_lengths[curr_dist_vec_num]  # much faster than using np.linalg.norm()
        #     if curr_step != step_size:
        #         if dist_vec_length - curr_step > -0.00001:  # error for abandoned and acknowledged was here
        #             num_steps += 1
        #             curr_pos = curr_pos + dist_vec / dist_vec_length * curr_step
        #             # calculate new distance vector
        #             dist_vecs[curr_dist_vec_num] = letter_points[curr_pos_num + 1] - curr_pos
        #             step_points.append((curr_pos + np.array([0.5, 0.5])) / self.keyboardLength)
        #             curr_step = step_size
        #         else:
        #             curr_step -= dist_vec_length
        #             curr_dist_vec_num += 1
        #             curr_pos_num += 1
        #             curr_pos = letter_points[curr_pos_num]

        #     elif int(dist_vec_length / step_size + 0.00001) > 0:  # adding 0.00001 to avoid rounding errors
        #         num_points_on_line = int(dist_vec_length / step_size + 0.00001)
        #         num_steps += num_points_on_line
        #         for i in range(0, num_points_on_line):
        #             step_points.append(((curr_pos + (i + 1) * (dist_vec / dist_vec_length * step_size)) + np.array(
        #                 [0.5, 0.5])) / self.keyboardLength)

        #         if dist_vec_length - num_points_on_line * step_size > 0.00001:
        #             curr_step -= (dist_vec_length - num_points_on_line * step_size)

        #         curr_dist_vec_num += 1
        #         curr_pos_num += 1
        #         curr_pos = letter_points[curr_pos_num]

        #     else:
        #         curr_step -= dist_vec_length
        #         curr_dist_vec_num += 1
        #         curr_pos_num += 1
        #         curr_pos = letter_points[curr_pos_num]

        step_points_normalized = self.normalize(step_points, 2)
        return step_points, step_points_normalized

    def get_word_graph_step_point_direct(self, word, letter_pos):
        no_duplicates_word = ""
        for letter in word:
            if letter not in self.availableChars:
                return None, None
            if (no_duplicates_word == "" or no_duplicates_word[len(no_duplicates_word) - 1] != letter):
                no_duplicates_word += letter

        word = no_duplicates_word

        letter_points = self.get_points_for_word(word, letter_pos)
        step_points = []
        for point in letter_points:
            step_points.append((point + np.array([0.5, 0.5])) / self.keyboardLength)

        # length = self.get_length_by_points(letter_points)
        # if length == 0:
        #     step_points = []
        #     curr_pos = letter_points[0]
        #     for i in range(0, steps):
        #         step_points.append((curr_pos + np.array([0.5, 0.5])) / self.keyboardLength)
        #     step_points_normalized = self.normalize(step_points, 1)
        #     return step_points, step_points_normalized

        # t = WordGraphGenerator.get_length_from_start_normalised(letter_points)
        # letter_points = np.array(letter_points)
        # x, y = letter_points[:,0], letter_points[:,1]
        # #print(x)
        # #print(y)
        # #print(t)
        # spl = scipy.interpolate.make_interp_spline(t, np.c_[x, y])

        # t_new = np.linspace(0, 1, steps)
        # x_new, y_new = spl(t_new).T

        # #print(x_new)
        # #print(y_new)
        # step_points = (np.c_[x_new, y_new] + 0.5) / self.keyboardLength
        #print(step_points)
        step_points_normalized = self.normalize(step_points, 2)

        return step_points, step_points_normalized


    # Normalizes the points according to the paper talking about SHARK2 (make all bounding boxes of shapes equally
    # big and put the center to the (0,0) point)
    # "letterpoints": np.array list, points to normalize
    # "length": int, length the longest side of the bounding box will have
    def normalize(self, letter_points, length):        
        (x, y) = self.get_xy(letter_points)
        #print("xy")
        #print(x)
        #print(y)
        bounding_box = [min(x), max(x), min(y), max(y)]
        bounding_box_size = [max(x) - min(x), max(y) - min(y)]

        if max(bounding_box_size[0], bounding_box_size[1]) != 0:
            s = length / max(bounding_box_size[0], bounding_box_size[1])
        else:
            s = 1

        middle_point = np.array([(bounding_box[0] + bounding_box[1]) / 2, (bounding_box[2] + bounding_box[3]) / 2])

        new_points = []
        for point in letter_points:
            new_points.append((point - middle_point) * s)

        return new_points

    # Reads the file that contains all available layouts (does not return anything, but assigns things to the
    # "layout" and "layoutKeys").
    def read_layouts_file(self, layouts_file_path):
        layout_name = None
        padding = []
        keys = []

        with open(layouts_file_path, "r", encoding="utf-8") as f:
            # Skip the first 6 header lines
            for _ in range(6):
                next(f)

            for line in f:
                if layout_name is None:
                    layout_name = line.rstrip()
                    self.layouts.append(layout_name)
                elif line.rstrip() == "-----":
                    self.layoutKeys[layout_name] = (padding, keys)
                    padding = []
                    keys = []
                    layout_name = None
                else:
                    splits = line.rstrip().split("$$")
                    if len(splits) > 1 and splits[1]:
                        padding.append(float(splits[1]))
                    else:
                        padding.append(0)
                    keys.append(splits[0])
            print("available layouts: ", self.layouts)

        available_chars = ""
        for i in range(0, len(self.layoutKeys[self.layout][1])):
            for character in self.layoutKeys[self.layout][1][i]:
                if character != " " and character != "<":
                    available_chars += character.lower()
        self.availableChars = available_chars

    @staticmethod
    def get_xy(word_points):
        x_points = []
        y_points = []
        for i in range(0, len(word_points)):
            x_points.append(word_points[i][0])
            y_points.append(word_points[i][1])
        return x_points, y_points


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('layouts_file', help='The txt file containing the layout specifications.')
    parser.add_argument('lexicon_file',
                        help='The txt file containing the frequency sorted word list to use as lexicon.')
    parser.add_argument('layout', help='The name of the layout specified in the layouts file to generate graphs for.')
    parser.add_argument('output_directory', help='The directory into which to write the output graph files.')
    parser.add_argument("--curve", nargs="?", required=False, const="line", type=str, help="'line', 'spline', 'direct', 'beziercubic'")
    parser.add_argument("--samples", nargs="?", required=False, const="20", type=int)

    args = parser.parse_args()

    layout = args.layout
    lexicon_file_path = args.lexicon_file
    layouts_file_path = args.layouts_file
    output_dir = args.output_directory
    curve = args.curve
    samples = args.samples
    if (samples < 20 and curve != 'direct'):
        print("There must be at least 20 samples!")
        return

    with open(lexicon_file_path, "r", encoding="utf-8") as lexicon_file:
        lexicon = [line.rstrip() for line in lexicon_file]

    wsg = WordGraphGenerator(layout, layouts_file_path)

    # Add all characters from the keyboard layout to the lexicon, such that the user will be able to use these
    # characters later on.
    for char in wsg.availableChars:
        if char not in lexicon:
            lexicon.append(char)

    start_time = time.time()
    with open(os.path.join(output_dir, f'graph_{layout}_{curve}_{samples}.txt'), 'w') as graph_file:
        char_pos = wsg.get_character_positions()
        #n = 1
        for word in lexicon:
            #print(f"Word #{n} started!")
            #n += 1
            if (curve == 'line'):
                #graph_points, graph_points_normalized = wsg.get_word_graph_step_point(word, samples, char_pos)
                graph_points, graph_points_normalized = wsg.get_word_graph_step_bezier_curve(word, samples, char_pos, [])
            elif (curve == 'spline'):
                graph_points, graph_points_normalized = wsg.get_word_graph_step_point_splines(word, samples, char_pos)
            elif (curve == 'direct'):
                graph_points, graph_points_normalized = wsg.get_word_graph_step_point_direct(word, char_pos)
            elif (curve == 'bezierquadratic'):
                graph_points, graph_points_normalized = wsg.get_word_graph_step_bezier_curve(word, samples, char_pos, [5.508e-01, -1.558e-02])                
            elif (curve == 'beziercubic'):
                graph_points, graph_points_normalized = wsg.get_word_graph_step_bezier_curve(word, samples, char_pos, [0.2913, -0.0143, 0.7848, 0.0420])
            else:
                print(f"No curve type = '{curve}' exists!")
                return 

            if graph_points is None:
                #print(f"Warning: No graph could be generated for the word '{word}'")
                # There is a letter in the word, that is not on the keyboard and therefore no graph can be generated for
                # this word and layout
                continue

            graph_points_new = []
            for point in graph_points:
                graph_points_new.append(round(point[0], 5))
                graph_points_new.append(round(point[1], 5))

            graph_points_normalized_new = []
            for point in graph_points_normalized:
                graph_points_normalized_new.append(round(point[0], 5))
                graph_points_normalized_new.append(round(point[1], 5))
            graph_file.write(word + ":")

            k = 0
            graph_length = len(graph_points_new)
            for i in graph_points_new:
                k += 1
                graph_file.write(str(i))
                if k < graph_length:
                    graph_file.write(",")
            graph_file.write(":")

            k = 0
            graph_length = len(graph_points_normalized_new)
            for i in graph_points_normalized_new:
                k += 1
                graph_file.write(str(i))
                if k < graph_length:
                    graph_file.write(",")
            graph_file.write("\n")
        graph_file.close()

    print(time.time() - start_time)


if __name__ == "__main__":
    main()
