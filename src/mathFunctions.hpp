#pragma once
#include <cmath>

class mathFunctions {
public:    
    static double sigmoidFunction(double inputVal) {
        if (inputVal >= 0.0) {
            const double expValue = std::exp(-inputVal);
            return 1.0 / (1.0 + expValue);
        }
        const double expValue = std::exp(inputVal);
        return expValue / (1.0 + expValue);
    }

    static double tanhFunction(double inputVal) {
        return std::tanh(inputVal);
    }
};