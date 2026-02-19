#ifndef UTILITIES_H
#define UTILITIES_H

#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <json/json.h>
#include <json/value.h>
#ifdef USE_AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <lgmath/CommonMath.hpp>
using DTYPE = autodiff::real1st;
#else
using DTYPE = double;
#endif

#include "spacetime/logger.hpp"

#define MDEG2RAD 0.0000174532925
#define TOLERANCE 1e-8

template <typename T = double>
const Eigen::MatrixX<T> ZERO(int rows) { return Eigen::MatrixX<T>::Zero(rows, rows); }
template <typename T = double>
const Eigen::MatrixX<T> EYE(int rows) { return Eigen::MatrixX<T>::Identity(rows, rows); }

// Throws an error if the rotation matrix is not orthonormal
//
// input:
//   C: a 3x3 matrix
//
// From: Timothy D Barfoot and Paul T Furgale,
//       Associating Uncertainty with Three-Dimensional Poses for use in Estimation Problems
//		 DOI: 10.1109/TRO.2014.2298059
template <typename Derived>
void validateRotationMatrix(const Eigen::MatrixBase<Derived> &C)
{
    assert((C.cols() == 3 && C.rows() == 3) && "Input needs to be 3x3 matrix");

    const Eigen::Matrix<typename Derived::Scalar, 3, 3> CtC = C.transpose() * C;
    Eigen::Matrix<typename Derived::Scalar, 3, 3> E = CtC - Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
    E = E.array().abs();

    double error = E.maxCoeff();

    assert((error < 1e-5) && "C must be a valid rotation matrix");
}

// Throws an error if the transformation matrix is not in SE(3)
//
// input:
//   T: a 4x4 matrix
//
// From: Timothy D Barfoot and Paul T Furgale,
//       Associating Uncertainty with Three-Dimensional Poses for use in Estimation Problems
//		 DOI: 10.1109/TRO.2014.2298059
template <typename Derived>
void validateTransformationMatrix(const Eigen::MatrixBase<Derived> &T)
{
    assert((T.rows() == 4 && T.cols() == 4) && "Input must be a 4x4 matrix");

    const auto C = T.template block<3, 3>(0, 0);
    validateRotationMatrix(C);

    Eigen::Matrix<typename Derived::Scalar, 1, 4> bottom_T;
    bottom_T << 0, 0, 0, 1;

    Eigen::Matrix<typename Derived::Scalar, 1, 4> diff = bottom_T - T.bottomRows(1);
    diff = diff.array().abs();

    double error = diff.maxCoeff();
    assert((error < 1e-10) && "T must be a valid transformation matrix");
}

void validateDiagonalMatrix(const Eigen::MatrixXd &matrix);

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 4> invertTransformation(const Eigen::MatrixBase<Derived> &mat)
{
    validateTransformationMatrix(mat); // Can remove (unsafe)

    Eigen::Matrix<typename Derived::Scalar, 4, 4> inv_T = Eigen::Matrix<typename Derived::Scalar, 4, 4>::Identity();
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R_transpose = mat.template topLeftCorner<3, 3>().transpose();

    inv_T.template block<3, 3>(0, 0) = R_transpose;
    inv_T.template block<3, 1>(0, 3) = -R_transpose * mat.template block<3, 1>(0, 3);

    return inv_T;
}

template <typename Derived>
void validateDiagonalMatrix(const Eigen::MatrixBase<Derived> &matrix)
{
    assert((matrix.rows() == matrix.cols()) && "Input needs to be square matrix");

    bool is_diagonal = true;
    for (unsigned int i = 0; i < matrix.rows(); i++)
    {
        for (unsigned int j = 0; j < matrix.cols(); j++)
        {
            if (i != j && abs(matrix(i, j)) > 1e-10)
            {
                is_diagonal = false;
            }
        }
    }

    assert(is_diagonal && "Matrix can only have non-zero entries on diagonal");
}

template <typename Derived>
Eigen::MatrixX<typename Derived::Scalar> invertDiagonal(const Eigen::MatrixBase<Derived> &matrix)
{
    validateDiagonalMatrix(matrix); // Can remove (unsafe)

    Eigen::MatrixX<typename Derived::Scalar> matrix_inv = matrix;

    for (unsigned int i = 0; i < matrix_inv.rows(); i++)
    {
        matrix_inv(i, i) = 1 / matrix_inv(i, i);
    }

    return matrix_inv;
}

template <typename M>
M loadCsv(const std::string &path)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
}

template <typename Derived>
bool loadArrayData(Json::Value &root, const std::string &key, Derived &out)
{
    if (root.isMember(key))
    {
        std::vector<typename Derived::Scalar> data;
        for (unsigned int i = 0; i < root[key].size(); i++)
        {
            double value = root[key][i].asDouble();
            data.push_back(static_cast<typename Derived::Scalar>(value));
        }
        if (Derived::RowsAtCompileTime == Eigen::Dynamic || Derived::ColsAtCompileTime == Eigen::Dynamic)
        {
            out.resize(root[key].size(), 1);
        }
        out = Eigen::Map<Derived>(data.data(), out.rows(), out.cols());
        info() << "Loaded array data with key: " << key;
        return true;
    }
    else
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
}

template <typename Derived>
bool loadMatrix4Data(Json::Value &root, const std::string &key, Derived &out)
{
    if (root.isMember(key))
    {
        std::vector<typename Derived::Scalar> data;
        for (unsigned int i = 0; i < root[key].size(); i++)
        {
            double value = root[key][i].asDouble();
            data.push_back(static_cast<typename Derived::Scalar>(value));
        }
        out = Eigen::Map<Derived>(data.data()).transpose();
        info() << "Loaded 4x4 array data with key: " << key;
        return true;
    }
    else
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
}

template <typename T = double>
void saveArrayData(Json::Value &out, const Eigen::VectorXd &in)
{
    out.clear();
    for (unsigned int i = 0; i < in.size(); i++)
    {
        out.append(in(i));
    }
}

template <typename T = double>
void saveMatrix4dData(Json::Value &out, const Eigen::Matrix4d &in)
{
    out.clear();
    for (unsigned int i = 0; i < 4; i++)
    {
        for (unsigned int j = 0; j < 4; j++)
        {
            out.append(in(j, i));
        }
    }
}

template <typename T>
T stoX(std::string value)
{
    if constexpr (std::is_same<T, int>::value)
        return std::stoi(value);
    else if constexpr (std::is_same<T, double>::value)
        return std::stod(value);
    else if constexpr (std::is_same<T, float>::value)
        return std::stof(value);
    else if constexpr (std::is_same<T, long>::value)
        return std::stol(value);
    else if constexpr (std::is_same<T, unsigned long>::value)
        return std::stoul(value);
    else if constexpr (std::is_same<T, long long>::value)
        return std::stoll(value);
    else if constexpr (std::is_same<T, unsigned long long>::value)
        return std::stoull(value);
    else if constexpr (std::is_same<T, std::string>::value)
        return value; // No conversion needed for string

    throw std::runtime_error("Unsupported type");
    return T();
}

class CSVReader
{
public:
    CSVReader(std::string file_path) : m_stream(file_path) {};

    bool burnLine();

    template <typename T = double>
    bool readLine(std::vector<T> &values)
    {
        std::string row;
        std::getline(m_stream, row, '\n');

        values.clear();
        auto start = 0U;
        auto end = row.find(',');
        while (end != std::string::npos)
        {
            values.push_back(stoX<T>(row.substr(start, end - start)));
            start = end + 1;
            end = row.find(',', start);
        }
        end = row.length(); // Handle last character when rows don't end in comma
        std::string substr = row.substr(start, end - start);
        substr.erase(std::remove(substr.begin(), substr.end(), ' '), substr.end());
        if (substr.length() > 0)
        {
            values.push_back(stoX<T>(row.substr(start, end - start)));
        }

        return values.size() > 0;
    }

    template <typename T = unsigned long>
    bool read(T &value)
    {
        std::string cell = "";
        std::getline(m_stream, cell, ',');

        if (cell.size() == 0)
            return false;
        value = stoX<T>(cell);
        return true;
    }

    void close();

private:
    std::ifstream m_stream;
};

template <typename Derived>
void printSparsity(const Eigen::MatrixBase<Derived> &A, std::string label, int row_size, int col_size)
{
    std::cout << std::endl
              << "Sparsity pattern of " << label << " (each entry is a " << row_size << "x" << col_size << " block):" << std::endl;
    for (unsigned int i = 0; i < A.rows() / row_size; i++)
    {
        for (unsigned int j = 0; j < A.cols() / col_size; j++)
        {
            Eigen::MatrixX<typename Derived::Scalar> block = A.block(row_size * i, col_size * j, row_size, col_size);

            if (block.hasNaN())
            {
                std::cout << "N ";
            }
            else if (block.isZero())
            {
                std::cout << "- ";
            }
            else
            {
                std::cout << "X ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename Derived>
void printSparsity(const Eigen::MatrixBase<Derived> &A, std::string label = "-", int block_size = 6)
{
    printSparsity(A, label, block_size, block_size);
}

Eigen::SparseMatrix<double> getFullRankProjection(const Eigen::MatrixX<double> &A);

namespace lgmath::se3
{
    Eigen::Matrix<DTYPE, 6, 6> getdJdt(const Eigen::Matrix<DTYPE, 18, 1> &gamma, bool dt = true, int num_terms = 10);
}


template <typename T>
bool safeRead(Json::Value &root, const std::string &key, T &out)
{
    if (root.isMember(key))
    {
        if constexpr (std::is_same_v<T, Json::Value>)
        {
            out = root[key];
            info() << "Loaded sub-root with key: " << key;
            return true;
        }
        else
        {
            out = root[key].as<T>();
            info() << "Loaded values from key: " << key << " with value: " << out;
            return true;
        }
    }
    else
    {
        warning() << "Key '" << key << "' not found in JSON. Using default value: " << out;
        return false;
    }
}

#endif // UTILITIES_H
