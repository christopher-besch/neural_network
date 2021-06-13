#pragma once
#include "pch.h"

#include "sigmoid.h"

#include <cmath>

class Cost
{
public:
    virtual ~Cost() = default;
    // only for test data
    // return cost associated with an output <a> and desired output <y>
    virtual float fn(const arma::fvec& a, const arma::fvec& y) = 0;
    // return vector of partial derivatives \partial C_x / \partial z of output layer times sigmoid_prime(z)
    // -> error delta from output layer
    // one column per data set
    virtual arma::fmat error(const arma::fmat& z, const arma::fmat& a, const arma::fmat& y) = 0;

    virtual std::string to_str() = 0;

    static std::shared_ptr<Cost> get(const std::string& name);
};

class QuadraticCost : public Cost
{
    virtual float fn(const arma::fvec& a, const arma::fvec& y) override
    {
        // euclidean distance to perfect result = null vector
        return arma::norm(0.5f * arma::square(a - y));
    }
    virtual arma::fmat error(const arma::fmat& z, const arma::fmat& a, const arma::fmat& y) override
    {
        return (a - y) % sigmoid_prime(z);
    }

    virtual std::string to_str() override
    {
        return "quadratic";
    }
};

class CrossEntropyCost : public Cost
{
    virtual float fn(const arma::fvec& a, const arma::fvec& y) override
    {
        arma::fvec result = -y % arma::log(a) - (1 - y) % arma::log(1 - a);
        // take care of log of numbers close to 0
        result.replace(arma::datum::nan, 0.0f);
        // sum of all rows
        return arma::sum(result);
    }
    virtual arma::fmat error(const arma::fmat& z, const arma::fmat& a, const arma::fmat& y) override
    {
        return a - y;
    }
    virtual std::string to_str() override
    {
        return "cross_entropy";
    }
};
