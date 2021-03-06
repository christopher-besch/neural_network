#include "costs.h"

#include "pch.h"

namespace NeuralNet {
std::shared_ptr<Cost> Cost::get(const std::string& name) {
    if(name == "quadratic")
        return std::make_shared<QuadraticCost>();
    else if(name == "cross_entropy")
        return std::make_shared<CrossEntropyCost>();
    else
        raise_critical("Unable to find cost function with name '{}", name);
}
} // namespace NeuralNet
