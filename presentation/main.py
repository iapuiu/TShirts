import numpy as np
from knapsackSolver import knapsack


def analyzeColors(data_base):
    print(data_base)
    min_row = np.min(data_base, axis=0)
    print(min_row)

    Base_Black = []
    Base_Black.append([0, 0, 2, 0, 0, 0])
    Base_Black.append([0, 0, 0, 2, 0, 0])
    Base_Black.append([0, 0, 0, 0, 2, 0])

    Base_Blue = []
    Base_Blue.append([0, 0, 0, 2, 0, 0])
    Base_Blue.append([0, 0, 1, 0, 1, 0])
    Base_Blue.append([0, 0, 2, 0, 0, 0])

    Base_Red = []
    Base_Red.append([0, 0, 0, 1, 2, 0])
    Base_Red.append([0, 0, 0, 3, 0, 0])
    Base_Red.append([0, 1, 1, 0, 0, 0])

    Base_Green = []
    Base_Green.append([1, 1, 1, 3, 1, 1])
    Base_Green.append([1, 1, 2, 2, 1, 1])
    Base_Green.append([1, 1, 1, 2, 2, 1])

    Minimum_subtracted = data_base - min_row
    Bases = [[], [], [], []]
    freq = [{} for i in range(4)]
    for row_idx in range(data_base.shape[0]):
        row = data_base[row_idx]
        print(row)
        for color_idx in range(4):
            idx_start = 6*color_idx
            color_part = row[range(idx_start, idx_start+6)]
            color_part = np.array(color_part)
            print(Bases[color_idx])
            if(sum(color_part) > 0):
                if(str(color_part) in Bases[color_idx]):
                    print("Already found: ", color_part)
                else:
                    Bases[color_idx].append(str(color_part))
                    print("Found: ", color_part)
                if(str(color_part) in freq[color_idx].keys()):
                    freq[color_idx][str(color_part)] = freq[color_idx][str(color_part)] + 1
                else:
                    freq[color_idx][str(color_part)] = 1
    for color_idx in range(4):
        print("Color idx = {}".format(color_idx))
        for entry in Bases[color_idx]:
            print(entry)

    uniqrows = []
    for idx in range(data_base.shape[0]):
        row = str(data_base[idx])
        if(not(row in uniqrows)):
            uniqrows.append(row)
    print(len(uniqrows))

    for color_idx in range(4):
        print("\n\nColor = {}".format(color_idx))
        for color_part in freq[color_idx].keys():
            print(color_part + "\t" + str(freq[color_idx][color_part]))


def sendBoxesMaximally(box):
    pass


def evaluateBoxValue(F_goal, F_limit, box):
    # Compute how often we can send the box:
    counter_goal = 0
    counter_limit = 0
    counter_limit_violation = 0
    alpha = np.zeros(F_goal.shape[0])
    for row_idx in range(F_goal.shape[0]):
        for muliplyer in range(1, 2):
            if(np.all(F_goal[row_idx] >= muliplyer * box)):
                counter_goal = counter_goal + 1
                alpha[row_idx] = muliplyer
        if(np.all(F_limit[row_idx] >= box)):
            counter_limit = counter_limit + 1
        #if(np.any(F_limit[row_idx]*alpha[row_idx] < box)):
        #    counter_limit_violation = counter_limit_violation + 1

    res = counter_goal * np.sum(box)
    if(np.sum(box) == 10):
        res = counter_limit * np.sum(box)
    if(counter_limit_violation > 0):
        res = -1000
    return res, alpha


def greedySelectBox(data):
    print(data)
    freq = {}
    for rowidx in range(data.shape[0]):
        row = data[rowidx]
        for color in range(len(row)):
            if(row[color] > 0):
                key = (row[color], color)
                if(key in freq.keys()):
                    freq[key] += 1
                else:
                    freq[key] = 1

    for key in freq.keys():
        print(str(key) + "\t->\t" + str(freq[key]))
    # Find largest possible box that can be send out to the largest number of
    # stores
    # items = []
    # for key in freq.keys():
    #    color = key[1]
    #    fill = key[0]
    #    occurences = freq[key]
    #    items.append((occurences, fill))
    # print(items)
    # val, newBox = knapsack(items, 10)
    # print(val, newBox)


def computeNewBox(F_goal, F_limit):
    # Compute new box using dynamic programming:
    box_values = np.zeros((11, 25))
    box_fills = [[np.zeros(24) for i in range(25)] for j in range(11)]
    alpha_values = [[np.zeros(F_goal.shape[0]) for i in range(25)] for j in range(11)]
    for max_weight in range(1, 11):
        for max_shirt_type in range(1, 25):
            # Know that we may add at most
            # max(F_limit, down)[max_shirt_type] shirts of type MST
            L = np.max(F_limit, axis = 0)[max_shirt_type - 1]
            L = int(L)
            best_val = -2000.0
            best_box = 0.0*np.arange(24)
            best_alpha = 0.0*np.arange(F_goal.shape[0])
            for num_shirts_added in range(L+1):
                # Pretend we are adding num_shirts_added shirts of type MST
                w = int(max_weight - num_shirts_added)
                if(w >= 0):
                    if(max_shirt_type > 0):
                        box = box_fills[w][max_shirt_type - 1]
                        new_box = np.copy(box)
                        new_box[max_shirt_type-1] = num_shirts_added
                        value, alpha = evaluateBoxValue(F_goal, F_limit, new_box)
                        if(value > best_val):
                            best_box = np.copy(new_box)
                            best_val = value
                            best_alpha = alpha
            box_values[max_weight, max_shirt_type] = best_val
            box_fills[max_weight][max_shirt_type] = best_box
            alpha_values[max_weight][max_shirt_type] = best_alpha

    # Compute the best value
    best_box = []
    best_val = -100
    best_alpha = []
    for fill in [4, 6, 8, 10]:
        for i in range(25):
            if(box_values[fill, i] > best_val):
                box = box_fills[fill][i]
                if(np.sum(box) in [4, 6, 8, 10]):
                    best_val = box_values[fill, i]
                    best_box = box_fills[fill][i]
                    best_alpha = alpha_values[fill][i]
    # ind = np.unravel_index(np.argmax(box_values, axis=None), box_values.shape)
    # print(ind)
    # best_box = box_fills[ind[0]][ind[1]]
    # alpha = alpha_values[ind[0]][ind[1]]
    # best_value = box_values[ind]
    return [best_box, best_val, best_alpha]


def main():
    # box_list = []
    data = np.genfromtxt('./BaseData.csv', delimiter=',')
    # for i in range(25):
    #     box, val, ala = computeNewBox(data, data)
    #     sub_mtx = np.outer(ala, box)
    #     data = data - sub_mtx
    #     box_list.append(box)
    #     print(np.sum(data), np.sum(box))
    # for row in data:
    #     print(row)
    # print("----------")
    # for box in box_list:
    #     print(box)
    analyzeColors(data)

if __name__ == "__main__":
    main()
