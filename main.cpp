#include "neural_network.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <random>
__attribute__((unused)) static float relu(float f) { return std::max(f, 0.0f); }
__attribute__((unused)) static float derive_relu(float f) { return (f > 0 ? f : 0); }

__attribute__((unused)) static float tanh_(float f) { return std::tanh(f); }
__attribute__((unused)) static float derive_tanh(float f) {
    return 1.0f - std::pow(std::tanh(f), 2);
}

int main(int, char**) {
    std::ios::sync_with_stdio(false);
    Parser parser;
    using namespace std;
    // MNIST数据集
    auto train_x = parser.parse_features("../data/train-images-idx3-ubyte");
    auto train_y = parser.parse_labels("../data/train-labels-idx1-ubyte");
    auto test_x = parser.parse_features("../data/t10k-images-idx3-ubyte");
    auto test_y = parser.parse_labels("../data/t10k-labels-idx1-ubyte");
    //参数{28 * 28, 50, 10}表示构建一个三层的神经网络，其中
    //输入层是28*28=784个节点，隐层是50个节点，输出层是10个节点，隐层数量可以增加，第一个和最后一个总是表示输入层和输出层的节点数量
    NeuralNetwork nn({28 * 28, 50, 10}, 0.005, 30, 20, &relu, &derive_relu);
    // NeuralNetwork nn({28 * 28, 150, 10}, 0.005, 30, 20);
    nn.train(train_x, train_y, test_x, test_y);

    return 0;
}
