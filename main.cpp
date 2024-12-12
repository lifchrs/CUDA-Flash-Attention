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

    Matrix(size_t h, size_t w, float val) : height(h), width(w)
    {
        data = new float[height * width];
        std::fill(data, data + (height * width), val);
    }

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

void makeMatrix(const int width, const int height, const float val){

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

    Matrix O = Matrix(N, d, 0)
    // Matrix m = Matrix(1, N, -std::numeric_limits<float>::infinity())
    // Matrix l = Matrix(1, N, 0)


    const int B_c = std::stoi(argv[4]);
    const int B_r = std::stoi(argv[5]);

    const dim3 grid_dim(std::stoi(argv[6]), std::stoi(argv[7]), std::stoi(argv[8]));
    const dim3 block_dim(std::stoi(argv[9]), std::stoi(argv[10]), std::stoi(argv[11]));

    serial_flash_attn_kernel<<<grid_dim, block_dim, sram_size>>>(N, d, Q.data, K.data, V.data, B_c, B_r, Tc, Tr, O.data);

    return 0;