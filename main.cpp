#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

struct Matrix
{
    float *data;
    size_t height;
    size_t width;

    ~Matrix()
    {
        delete[] data;
    }
};

Matrix readMatrixFromFile(const std::string &fileName)
{
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    std::vector<std::vector<float>> tempMatrix;
    std::string line;
    size_t width = 0;

    while (std::getline(file, line))
    {
        std::istringstream stream(line);
        std::vector<float> row;
        float value;
        while (stream >> value)
        {
            row.push_back(value);
        }

        if (tempMatrix.empty())
        {
            width = row.size();
        }
        else if (row.size() != width)
        {
            throw std::runtime_error("Inconsistent row width in matrix file");
        }

        tempMatrix.push_back(row);
    }

    size_t height = tempMatrix.size();

    float *data = new float[height * width];

    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            data[i * width + j] = tempMatrix[i][j];
        }
    }

    return {data, height, width};
}

void printMatrix(const Matrix &matrix)
{
    for (size_t i = 0; i < matrix.height; ++i)
    {
        for (size_t j = 0; j < matrix.width; ++j)
        {
            std::cout << matrix.data[i * matrix.width + j] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    try
    {
        const std::string fileName = "12x10.txt";
        Matrix matrix = readMatrixFromFile(fileName);

        std::cout << "Matrix Dimensions: " << matrix.height << "x" << matrix.width << "\n";
        printMatrix(matrix);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
