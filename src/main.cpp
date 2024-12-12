#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "flash_attention.cuh"


extern void forward(float* Q, float* K, float* V, const int B_c, const int B_r, const int grid_dim_x, const int grid_dim_y, const int grid_dim_z, const int block_dim_x, const int block_dim_y, const int block_dim_z, const int N, const int d, const int T_c, const int T_r, float* O);

struct Matrix
{
    float *data;
    int height;
    int width;

    Matrix(int h, int w, float val) : height(h), width(w)
    {
        data = new float[height * width];
        std::fill(data, data + (height * width), val);
    }

    Matrix(float* data, int h, int w): data(data), height(h), width(w){}

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
    int width = 0;

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

    int height = tempMatrix.size();

    float *data = new float[height * width];

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            data[i * width + j] = tempMatrix[i][j];
        }
    }

    return {data, height, width};
}

void printMatrix(const Matrix &matrix)
{
    for (int i = 0; i < matrix.height; ++i)
    {
        for (int j = 0; j < matrix.width; ++j)
        {
            std::cout << matrix.data[i * matrix.width + j] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 12) {
        std::cerr << "Usage: " << argv[0] 
                  << " <q_file> <k_file> <v_file> "
                  << "<B_c> <B_r> <grid_dim_x> <grid_dim_y> <grid_dim_z> <block_dim_x> <block_dim_y> <block_dim_z>" 
                  << std::endl;
        return 1;
    }

    std::string q_file = argv[1];
    Matrix Q = readMatrixFromFile(q_file);
    std::string k_file = argv[2];
    Matrix K = readMatrixFromFile(k_file);
    std::string v_file = argv[3];
    Matrix V = readMatrixFromFile(v_file);


    
    const int N = Q.height;
    const int d = Q.width;

    Matrix O = Matrix(N, d, 0);
    // Matrix m = Matrix(1, N, -std::numeric_limits<float>::infinity())
    // Matrix l = Matrix(1, N, 0)


    const int B_c = std::stoi(argv[4]);
    const int B_r = std::stoi(argv[5]);

    const int T_c = (N + B_c - 1) / B_c;
    const int T_r = (N + B_r - 1) / B_r;

    forward(Q.data, K.data, V.data, B_c, B_r, std::stoi(argv[6]), std::stoi(argv[7]), std::stoi(argv[8]), std::stoi(argv[9]), std::stoi(argv[10]), std::stoi(argv[11]), N, d, T_c, T_r, O.data);

    // std::cout << "hi";
    printMatrix(O);

    return 0;
}