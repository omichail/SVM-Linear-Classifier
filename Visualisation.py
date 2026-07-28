# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def read_points(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            x, y, label = map(float, line.strip().split())
            points.append((x, y, int(label)))
    return points

def read_svm_result(filename):
    with open(filename, 'r') as file:
        line = file.readline().strip()
        if line.startswith("The data is not linearly separable"):
            return None
        else:
            w1, w2, b = map(float, line.split())
            return (w1, w2, b)

def plot_points_and_svm(points, svm_result):
  
    red_points = [point for point in points if point[2] == -1]
    green_points = [point for point in points if point[2] == 1]

    red_x = [point[0] for point in red_points]
    red_y = [point[1] for point in red_points]
    green_x = [point[0] for point in green_points]
    green_y = [point[1] for point in green_points]

    plt.scatter(red_x, red_y, color='red')
    plt.scatter(green_x, green_y, color='green')
    
    if svm_result:
        w1, w2, b = svm_result
        x_vals = [min(x for x, y, label in points), max(x for x, y, label in points)]
        y_vals = [-(w1/w2) * x - b/w2 for x in x_vals]
        plt.plot(x_vals, y_vals, color='blue')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SVM Classification')
    plt.show()

def main():
    points = read_points('points.txt')
    svm_result = read_svm_result('result.txt')
    
    if svm_result:
        print(f"Linear decision boundary: {svm_result[0]} * x + {svm_result[1]} * y + {svm_result[2]} = 0")
    else:
        print("The data is not linearly separable.")
    
    plot_points_and_svm(points, svm_result)

if __name__ == "__main__":
    main()
