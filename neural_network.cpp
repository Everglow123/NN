//
// Created by zhouheng on 20-8-12.
//

#include "neural_network.h"
//#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
NeuralNetwork::NeuralNetwork(const std::vector<int>& shape, float learningRate, int batch_size,
                             int epoch, activate_func_t activateFunc,
                             derive_activate_func_t deriveActivateFunc)
    : layerCount_(shape.size()), learningRate_(learningRate), activateFunc_(activateFunc),
      deriveActivateFunc_(deriveActivateFunc), inputSize_(shape.front()), outputSize_(shape.back()),
      batchSize_(batch_size), epoch_(epoch) {
    for (int i = 0; i < layerCount_ - 1; ++i) {
        layers_.push_back(Layer(shape[i], shape[i + 1]));
        layers_.back().randomInit();
    }
    layers_.push_back(Layer(shape.back(), 1));
};
void NeuralNetwork::forwardPropagation(const Eigen::VectorXf& x) {
    using namespace std;
    using namespace Eigen;
    this->layers_.front().output = x / 255;
    for (int i = 1; i < layerCount_; ++i) {
        layers_[i].output = (layers_[i - 1].weights * layers_[i - 1].output) + layers_[i].bias;
        // cout << layers_[i].bias << endl;
        layers_[i].output = layers_[i].output.unaryExpr(this->activateFunc_);
        // cout << layers_[i].output << endl;
    }
};

void NeuralNetwork::backPropagation(const Eigen::VectorXf& y) {
    using namespace std;
    using namespace Eigen;
    this->layers_.back().loss = (y - this->layers_.back().output);
    VectorXf& lastOutput = this->layers_.back().output;  //输出层的输出

    Eigen::VectorXf temp = ((this->layers_.back().loss * (-2)).array() *
                            (lastOutput.unaryExpr(this->deriveActivateFunc_).array()))
                               .matrix();
    // cout << temp.rows() << endl;

    
    this->layers_.back().biasGrads = temp;
    this->layers_[layerCount_ - 2].weightGrads =
        temp * (this->layers_[layerCount_ - 2].output.transpose());
    this->layers_[layerCount_ - 2].loss = this->layers_[layerCount_ - 2].weights.transpose() * temp;
    for (int i = layerCount_ - 3; i >= 0; --i) {
        auto& layer = this->layers_[i];
        auto& nextLayer = this->layers_[i + 1];
        Eigen::VectorXf temp1 = ((nextLayer.loss).array() *
                                 (nextLayer.output.unaryExpr(this->deriveActivateFunc_).array()))
                                    .matrix();
        nextLayer.biasGrads = temp1;
        layer.weightGrads = temp1 * (layer.output.transpose());
        if (i != 0)
            layer.loss = layer.weights.transpose() * temp1;
    }
};
void NeuralNetwork::gradientDescent() {
    using namespace std;
    for (int i = 0; i < layerCount_ - 1; ++i) {
        auto& l = layers_[i];
        l.bias -= l.biasGrads * learningRate_;
        l.weights -= l.weightGrads * learningRate_;
    }
}

void NeuralNetwork::gradientClean() {
    for (int i = 0; i < layerCount_ - 1; ++i) {
        auto& l = layers_[i];
        l.weightGrads.setZero();
        if (i != 0)
            l.biasGrads.setZero();
    }
    layers_.back().biasGrads.setZero();
}
double NeuralNetwork::getLoss() { return this->layers_.back().loss.norm(); }
void NeuralNetwork::train(std::vector<std::vector<float>>& train_xs,
                          std::vector<std::vector<float>>& train_ys,
                          std::vector<std::vector<float>>& test_xs,
                          std::vector<std::vector<float>>& test_ys) {
    using namespace std;
    using namespace chrono;
    using namespace Eigen;
    assert(train_xs.size() == train_ys.size());
    assert(test_xs.size() == test_ys.size());
    ssize_t trainingDataSize = train_xs.size();
    ssize_t testDataSize = test_xs.size();
    vector<VectorXf> trainXvecs, trainYvecs, testXvecs, testYvecs;
    trainXvecs.reserve(trainingDataSize);
    trainYvecs.reserve(trainingDataSize);
    testXvecs.reserve(testDataSize);
    testYvecs.reserve(testDataSize);
    for (int i = 0; i < trainingDataSize; ++i) {
        trainXvecs.push_back(NeuralNetwork::stdVecToEigenVec(train_xs[i]));
        // xvecs.push_back(VectorXf(xs[i].data(),inputSize_));
    }
    for (int i = 0; i < trainingDataSize; ++i) {
        trainYvecs.push_back(NeuralNetwork::stdVecToEigenVec(train_ys[i]));
    }
    for (int i = 0; i < testDataSize; ++i) {
        testXvecs.push_back(NeuralNetwork::stdVecToEigenVec(test_xs[i]));
    }
    for (int i = 0; i < testDataSize; ++i) {
        testYvecs.push_back(NeuralNetwork::stdVecToEigenVec(test_ys[i]));
    }
    for (int e = 0; e < epoch_; ++e) {
        int i = 0;
        double trainingLoss = 0;
        auto t = system_clock::now();
        while (i < trainingDataSize) {
            this->forwardPropagation(trainXvecs[i]);
            this->backPropagation(trainYvecs[i]);
            this->gradientDescent();
            this->gradientClean();
            trainingLoss += this->getLoss();
            ++i;
        }
        // cout<<this->layers_[layerCount_-2].weights<<endl;
        // abort();
        cout << "第" << e << "轮已完成,耗时: " << fixed
             << duration<double>(system_clock::now() - t).count() << "秒,精度：";
        cout.flush();
        cout << fixed << this->getAccuracy(testXvecs, testYvecs) * 100 << R"(%,loss: )" << fixed
             << trainingLoss / trainingDataSize << endl;
    }
};
float NeuralNetwork::getAccuracy(std::vector<Eigen::VectorXf>& test_xvs,
                                 std::vector<Eigen::VectorXf>& test_yvs) {
    using namespace std;
    using namespace Eigen;
    assert(test_xvs.size() == test_yvs.size());
    ssize_t testDataSize = test_xvs.size();
    ssize_t correctCount = 0;
    for (int i = 0; i < testDataSize; ++i) {
        this->forwardPropagation(test_xvs[i]);
        VectorXf::Index predMaxRow, realMaxRow;
        this->layers_.back().output.maxCoeff(&predMaxRow);
        test_yvs[i].maxCoeff(&realMaxRow);
        if (predMaxRow == realMaxRow) {
            correctCount++;
        }
    }
    // cout << correctCount << endl;
    return float(correctCount) / float(testDataSize);
}
float NeuralNetwork::test(std::vector<std::vector<float>>& test_xs,
                          std::vector<std::vector<float>>& test_ys) {
    using namespace std;
    using namespace Eigen;
    assert(test_xs.size() == test_ys.size());
    vector<VectorXf> xvecs;
    xvecs.reserve(test_xs.size());
    vector<VectorXf> yvecs;
    yvecs.reserve(test_ys.size());

    ssize_t testDataSize = test_xs.size();
    ssize_t correctCount = 0;
    for (int i = 0; i < testDataSize; ++i) {
        xvecs.push_back(NeuralNetwork::stdVecToEigenVec(test_xs[i]));
    }
    for (int i = 0; i < testDataSize; ++i) {
        yvecs.push_back(NeuralNetwork::stdVecToEigenVec(test_ys[i]));
    }

    for (int i = 0; i < testDataSize; ++i) {
        this->forwardPropagation(xvecs[i]);
        VectorXf::Index predMaxRow, realMaxRow;
        this->layers_.back().output.maxCoeff(&predMaxRow);
        yvecs[i].maxCoeff(&realMaxRow);
        if (predMaxRow == realMaxRow) {
            correctCount++;
        }
    }
    // cout << correctCount << endl;
    return float(correctCount) / float(testDataSize);
}
