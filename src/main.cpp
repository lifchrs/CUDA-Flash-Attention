#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "flash_attention.cuh"
#include <limits>
#include <chrono>
#include <iomanip>

extern void forward_serial(
    float *Q_h,           // Query matrix
    float *K_h,           // Key matrix
    float *V_h,           // Value matrix
    float *O_h,           // Output matrix
    float *m_h,           // Running maximum values
    float *l_h,           // Running sum of exponentials
    const int B_c,        // Column block size
    const int B_r,        // Row block size
    const int batch_size, // Batch size
    const int num_heads,  // Number of heads
    const int seq_len,    // Sequence Length
    const int d,          // Hidden dimension
    const int block_dim_y // Inversely proportional to how many tiles a thread attends to
);

extern void forward_parallel(
    float *Q_h,           // Query matrix
    float *K_h,           // Key matrix
    float *V_h,           // Value matrix
    float *O_h,           // Output matrix
    float *m_h,           // Running maximum values
    float *l_h,           // Running sum of exponentials
    const int B_c,        // Column block size
    const int B_r,        // Row block size
    const int batch_size, // Batch size
    const int num_heads,  // Number of heads
    const int seq_len,    // Sequence Length
    const int d,          // Hidden dimension
    const int block_dim_y // Inversely proportional to how many tiles a thread attends to
);

// extern void forward(float *Q, float *K, float *V, const int B_c, const int B_r, const int grid_dim_x, const int grid_dim_y, const int grid_dim_z, const int block_dim_x, const int block_dim_y, const int block_dim_z, const int N, const int d, const int T_c, const int T_r, float *O);

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

    Matrix(float *data, int h, int w) : data(data), height(h), width(w) {}

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

void writeMatrixToFile(const Matrix &matrix, const std::string &fileName)
{
    const int max_precision = std::numeric_limits<float>::max_digits10;

    std::ofstream file(fileName);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    for (size_t i = 0; i < matrix.height; ++i)
    {
        for (size_t j = 0; j < matrix.width; ++j)
        {
            file << std::setprecision(max_precision) << std::fixed << matrix.data[i * matrix.width + j];
            if (j < matrix.width - 1)
            {
                file << " "; // Add space between elements in the same row
            }
        }
        file << "\n"; // Newline after each row
    }

    file.close();
}

void printMatrix(const Matrix &matrix)
{
    const int max_precision = std::numeric_limits<float>::max_digits10;
    for (int i = 0; i < matrix.height; ++i)
    {
        for (int j = 0; j < matrix.width; ++j)
        {
            std::cout << std::setprecision(max_precision) << std::fixed << matrix.data[i * matrix.width + j] << " ";
        }
        std::cout << "\n";
    }
}

void setVal(Matrix &matrix, const float val)
{
    for (int i = 0; i < matrix.height; ++i)
    {
        for (int j = 0; j < matrix.width; ++j)
        {
            matrix.data[i * matrix.width + j] = val;
        }
    }
}

void reset(Matrix &O, Matrix &m, Matrix &l)
{
    setVal(O, 0);
    setVal(m, -(std::numeric_limits<float>::infinity()));
    setVal(l, 0);
}

int main(int argc, char *argv[])
{
    // if (argc != 16)
    // {
    //     std::cerr << "Usage: " << argv[0]
    //               << " <q_file> <k_file> <v_file> "
    //               << "<B_c> <B_r> <grid_dim_x> <grid_dim_y> <grid_dim_z> <block_dim_x> <block_dim_y> <block_dim_z> "
    //               << " <o_file> <print o> <timing_runs> <warmup>"
    //               << std::endl;

    //     std::cerr << "Has " << argc << "args" << std::endl;
    //     return 1;
    // }

    const int num_runs = std::stoi(argv[1]);
    const int warmups = std::stoi(argv[2]);
    const int print_matrix = std::stoi(argv[3]);

    std::string q_file = argv[4];
    std::string k_file = argv[5];
    std::string v_file = argv[6];
    std::string o_file = argv[7];
    std::cerr << "files\n\n\n";
    for (int i = 4; i < 8; i++)
    {
        std::cerr << argv[i] << "\n";
    }

    // return;

    const int batch_size = std::stoi(argv[8]);
    const int num_heads = std::stoi(argv[9]);
    const int seq_len = std::stoi(argv[10]);
    const int emb_dim = std::stoi(argv[11]);

    const int B_c = std::stoi(argv[12]);
    const int B_r = std::stoi(argv[13]);
    const int block_dim_y = std::stoi(argv[14]);

    const int use_parallel = std::stoi(argv[15]);

    std::cerr << "a\n";
    Matrix Q = readMatrixFromFile(q_file);
    Matrix K = readMatrixFromFile(k_file);
    Matrix V = readMatrixFromFile(v_file);
    std::cerr << "b\n";

    Matrix O = Matrix(batch_size * num_heads * seq_len, emb_dim, 0);
    Matrix m = Matrix(batch_size * num_heads, seq_len, -(std::numeric_limits<float>::infinity()));
    Matrix l = Matrix(batch_size * num_heads, seq_len, 0);
    std::cerr << "c\n";

    auto forward = (use_parallel ? forward_parallel : forward_serial);

    for (int i = 0; i < warmups; ++i)
    {
        reset(O, m, l);
        forward(Q.data, K.data, V.data, O.data, m.data, l.data, B_c, B_r, batch_size, num_heads, seq_len, emb_dim, block_dim_y);
    }
    std::cerr << "d\n";

    long long total_time = 0;
    for (int i = 0; i < num_runs; ++i)
    {
        reset(O, m, l);
        std::cout << "timing" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        forward(Q.data, K.data, V.data, O.data, m.data, l.data, B_c, B_r, batch_size, num_heads, seq_len, emb_dim, block_dim_y);
        auto end = std::chrono::high_resolution_clock::now();

        total_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    std::cerr << "e\n";

    std::cout << static_cast<double>(total_time) / num_runs << std::endl;

    writeMatrixToFile(O, o_file);
    std::cerr << "f\n";

    if (print_matrix)
        printMatrix(O);

    return 0;
}