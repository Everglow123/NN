//
// Created by zhouheng on 20-8-12.
//

#ifndef NN_NEURALNETWORK_H_

#include <Eigen/Core>
#include <map>
#include <random>
#include <vector>

#include "layer.h"
class NeuralNetwork {
  public:
    using activate_func_t = float (*)(float);
    using derive_activate_func_t = activate_func_t;
    static float random_func(float f);
    explicit NeuralNetwork(
        const std::vector<int>& shape, float learning_rate = 0.01, int batch_size = 30,
        int epoch = 100, activate_func_t activate_func = NeuralNetwork::sigmoid,
        derive_activate_func_t deriveActivateFunc = NeuralNetwork::deriveSigmoid);

    void train(std::vector<std::vector<float>>& train_x, std::vector<std::vector<float>>& train_y,
               std::vector<std::vector<float>>& test_x, std::vector<std::vector<float>>& test_y);

    float test(std::vector<std::vector<float>>& x, std::vector<std::vector<float>>& y);

  private:
    float getAccuracy(std::vector<Eigen::VectorXf>& test_xvs,
                      std::vector<Eigen::VectorXf>& test_yvs);
    inline void forwardPropagation(const Eigen::VectorXf& x);

    inline void backPropagation(const Eigen::VectorXf& y);

    void gradientDescent();

    void gradientClean();
    int layerCount_;
    float learningRate_;

    double getLoss();
    float (*activateFunc_)(float);
    float (*deriveActivateFunc_)(float);
    int inputSize_;
    int outputSize_;
    int batchSize_;
    int epoch_;
    std::vector<Layer> layers_;
    static float sigmoid(float f) { return 1.0 / (1.0 + std::exp(-f)); }
    static float deriveSigmoid(float f) { return sigmoid(f) * (1 - sigmoid(f)); }

    inline Eigen::VectorXf stdVecToEigenVec(std::vector<float>& v) {
        //直接把std::vector 的数据转化成eigen的向量，没有额外的拷贝
        return Eigen::Map<Eigen::VectorXf>(v.data(), v.size());
    }
};

#define NN_NEURALNETWORK_H_

#endif  // NN_NEURALNETWORK_H_
