//
// Created by zhouheng on 20-8-11.
//

#include "utils.h"
#include <algorithm>
#include <cassert>
#include <memory>

std::vector<std::vector<float>> Parser::parse_features(const std::string& path) {
    using namespace std;
    vector<vector<float>> res;
    ifstream f(path, ios::binary);
    if (!f) {
        cerr << system("echo `pwd`") << endl;
        cerr << "unable to open: " << path << endl;
        abort();
    }
    char bytes[4];
    f.read(bytes, 4);
    f.read(bytes, 4);
    Parser::big_to_little(bytes);
    int count1 = *(reinterpret_cast<int*>(bytes));
    assert(count1 == 60000 || count1 == 10000);
    res.reserve(count1);
    f.read(bytes, 4);
    Parser::big_to_little(bytes);
    assert(*(reinterpret_cast<int*>(bytes)) == 28);
    f.read(bytes, 4);
    for (int i = 0; i < count1; ++i) {
        res.emplace_back();
        res.back().reserve(28 * 28);
        for (int j = 0; j < 28 * 28; ++j) {
            res.back().push_back(f.get());
        }
    }
    f.close();
    return res;
}

std::vector<std::vector<float>> Parser::parse_labels(const std::string& path) {
    using namespace std;
    std::vector<std::vector<float>> res;
    ifstream f(path, ios::binary);
    if (!f) {
        cerr << "unable to open: " << path << endl;
        return {};
    }
    char bytes[4];
    f.read(bytes, 4);
    f.read(bytes, 4);
    Parser::big_to_little(bytes);
    int count1;
    count1 = *(reinterpret_cast<int*>(bytes));

    assert(count1 == 60000 || count1 == 10000);
    res.reserve(count1);
    for (int i = 0; i < count1; ++i) {
        res.emplace_back(10, 0);
        res.back()[f.get()] = 1;
    }
    f.close();
    return res;
}

void Parser::big_to_little(char* bytes) {
    using namespace std;
    swap(bytes[0], bytes[3]);
    swap(bytes[1], bytes[2]);
}
