#ifndef LAYER_H_
#define LAYER_H_
#include <Eigen/Core>
#include <map>
#include <random>
#include <vector>
struct Layer {
    int size;
    int nextSize;

    Eigen::MatrixXf weights;      //权重
    Eigen::MatrixXf weightGrads;  //权重的梯度
    Eigen::VectorXf bias;         //偏置
    Eigen::VectorXf biasGrads;    //偏置的梯度
    Eigen::VectorXf output;       //这一层的输出
    Eigen::VectorXf loss;         //损失

    Layer() = default;

    //全零初始化
    Layer(int nodes, int nextNodes)
        : size(nodes), nextSize(nextNodes), weights(Eigen::MatrixXf::Zero(nextSize, nodes)),
          weightGrads(Eigen::MatrixXf::Zero(nextSize, size)), bias(Eigen::VectorXf::Zero(size)),
          biasGrads(Eigen::VectorXf::Zero(size)), output(Eigen::VectorXf::Zero(size)),
          loss(Eigen::VectorXf::Zero(size)){};

    //权重和偏置的正态分布随机初始化
    inline void randomInit() {
        using namespace std;

        // C++11的随机数
        random_device rd;
        default_random_engine e(rd());
        normal_distribution distribution(0.0, 0.01);

        // unaryExpr指对矩阵中的每一个元素执行操作
        weights = weights.unaryExpr([&](const float& f) -> float { return distribution(e); });
        bias = bias.unaryExpr([&](const float& f) -> float { return distribution(e); });
    };
};

#endif  // LAYER_H_
