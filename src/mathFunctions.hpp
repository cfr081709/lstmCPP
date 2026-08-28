#pragma once
#include <cmath>
#include <numbers>
#include <iostream>

using namespace std;

class mathFunctions {
public:    
    static double sigmoidFunction(double inputVal){
        double returnVal = 1 / (1 + pow(numbers::e, inputVal * -1));
        return returnVal;
    }

    static double tanhFunction(double inputVal){
        double numerator = (pow(numbers::e, inputVal)) - pow(numbers::e, inputVal * -1);
        double denominator = (pow(numbers::e, inputVal) + pow(numbers::e, inputVal * -1));
        return numerator / denominator;
    }
};