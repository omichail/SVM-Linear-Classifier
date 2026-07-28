#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <limits>

using namespace std;

struct Point
{
    double x, y;
    int label;
};

struct SVMResult
{
    double w1, w2, b;
    bool isLinearlySeparable;
};


SVMResult trainSVM(const vector<Point>& points)
{
    const double learningRate = 0.001;
    const double lambda = 0.01;
    const int maxIterations = 10000;

    double w1 = 0, w2 = 0, b = 0;

    for (int i = 0; i < maxIterations; i++)
    {
        for (const auto& point : points)
        {
            double decision = point.x * w1 + point.y * w2 + b;
            if (point.label * decision < 1)
            {
                w1 += learningRate * (point.label * point.x - 2 * lambda * w1);
                w2 += learningRate * (point.label * point.y - 2 * lambda * w2);
                b += learningRate * point.label;
            }
            else
            {
                w1 += learningRate * (-2 * lambda * w1);
                w2 += learningRate * (-2 * lambda * w2);
            }
        }
    }

    SVMResult result;
    result.w1 = w1;
    result.w2 = w2;
    result.b = b;
    result.isLinearlySeparable = true;

    for (const auto& point : points)
    {
        double decision = point.x * w1 + point.y * w2 + b;
        if (point.label * decision <= 0)
        {
            result.isLinearlySeparable = false;
            break;
        }
    }

    return result;
}

void savePointsAndResult(const vector<Point>& points, const SVMResult& svm, const string& pointsFilename, const string& resultFilename)
{
    ofstream pointsFile(pointsFilename);
    ofstream resultFile(resultFilename);

    if (!pointsFile || !resultFile)
    {
        cerr << "Error opening file for writing\n";
        return;
    }

    for (const auto& point : points)
    {
        pointsFile << point.x << " " << point.y << " " << point.label << endl;
    }

    if (svm.isLinearlySeparable)
    {
        resultFile << svm.w1 << " " << svm.w2 << " " << svm.b << endl;
    }
    else
    {
        resultFile << "The data is not linearly separable.\n";
    }
}


vector<Point> generatePoissonPoints(double intensity, double width, double height)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> xDist(0.0, width);
    uniform_real_distribution<> yDist(0.0, height);
    uniform_int_distribution<> labelDist(0, 1);

    double area = width * height;

    poisson_distribution<> poissonDist(intensity * area);
    int N = poissonDist(gen);

    vector<Point> points(N);
    for (int i = 0; i < N; ++i)
    {
        points[i].x = xDist(gen);
        points[i].y = yDist(gen);
        points[i].label = labelDist(gen) == 0 ? -1 : 1;
    }

    return points;
}

vector<Point> readPointsFromFile(const string& filename)
{
    vector<Point> points;
    ifstream infile(filename);
    if (!infile)
    {
        cerr << "Error reading file\n";
        return points;
    }

    int numPoints;
    infile >> numPoints;

    Point p;
    for (int i = 0; i < numPoints; ++i)
    {
        if (infile >> p.x >> p.y >> p.label)
        {
            points.push_back(p);
        }
        else
        {
            cerr << "Error reading point data\n";
            break;
        }
    }

    return points;
}


vector<Point> inputPointsManually(int N)
{
    vector<Point> points(N);
    cout << "Enter x, y, label (-1 for red, 1 for green): " << endl;
    for (int i = 0; i < N; i++)
    {
        cin >> points[i].x >> points[i].y >> points[i].label;
    }
    return points;
}

int main()
{
    int inputMethod;
    do {

        cout << "Choose input method (1 = manual, 2 = file, 3 = random): ";
        cin >> inputMethod;

        int N;

        vector<Point> points;

        if (inputMethod == 1)
        {
            cout << "Enter the number of points: ";
            cin >> N;
            points = inputPointsManually(N);
        }
        else if (inputMethod == 2)
        {
            points = readPointsFromFile("points1.txt");
        }
        else if (inputMethod == 3)
        {

            double intensity, width, height;

            cout << "Enter intensity: ";
            cin >> intensity;
            cout << "Enter width and height of the rectangle: ";
            cin >> width >> height;
            points = generatePoissonPoints(intensity, width, height);
        }
        else
        {
            cerr << "Invalid input method\n";
            return 1;
        }

        SVMResult svm = trainSVM(points);
        savePointsAndResult(points, svm, "points.txt", "result.txt");
        system("python Visualisation.py");
    } while (inputMethod >= 1 && inputMethod <= 3);

    return 0;
}
