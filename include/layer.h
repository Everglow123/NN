#ifndef LAYER_H_
#define LAYER_H_
#include <Eigen/Core>
#include <map>
#include <random>
#include <vector>
struct Layer {
    int size;
    int nextSize;
    Eigen::MatrixXf weights;
    Eigen::VectorXf output;
    Eigen::VectorXf bias;
    Eigen::MatrixXf weightGrads;
    Eigen::VectorXf loss;
    Eigen::VectorXf biasGrads;
    Layer() = default;
    Layer(int nodes, int nextNodes)
        : size(nodes), nextSize(nextNodes), weights(Eigen::MatrixXf::Zero(nextSize, nodes)),
          output(Eigen::VectorXf::Zero(size)), bias(Eigen::VectorXf::Zero(size)),
          weightGrads(Eigen::MatrixXf::Zero(nextSize, size)), loss(Eigen::VectorXf::Zero(size)),
          biasGrads(Eigen::VectorXf::Zero(size)){};

    inline void randomInit() {
        using namespace std;
        random_device rd;
        default_random_engine e(rd());
        normal_distribution distribution(0.0, 0.01);
        weights = weights.unaryExpr([&](const float& f) -> float { return distribution(e); });
        bias = bias.unaryExpr([&](const float& f) -> float { return distribution(e); });
    };
};

#endif  // LAYER_H_
