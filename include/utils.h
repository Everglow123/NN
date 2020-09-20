#ifndef UTILS_H_
#define UTILS_H_

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

class Parser {
public:

    std::vector<std::vector<float>> parse_features(const std::string &path);

    std::vector<std::vector<float>> parse_labels(const std::string &path);

private:

    static void big_to_little(char *bytes);
};


#endif  // UTILS_H_