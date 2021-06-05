#pragma once
#include "pch.h"

#include "sigmoid.h"

#include <cmath>

class Cost
{
public:
    // only for test data
    // return cost associated with an output <a> and desired output <y>
    virtual float fn(const arma::fvec& a, const arma::fvec& y) = 0;
    // return vector of partial derivatives \partial C_x / \partial z of output layer
    // one column per data set
    virtual arma::fmat error(const arma::fmat& z, const arma::fmat& a, const arma::fmat& y) = 0;

    virtual ~Cost() = default;
};

class QuadraticCost : public Cost
{
    virtual float fn(const arma::fvec& a, const arma::fvec& y) override
    {
        return arma::norm(0.5f * arma::square(a - y));
    }
    virtual arma::fmat error(const arma::fmat& z, const arma::fmat& a, const arma::fmat& y) override
    {
        return (a - y) * sigmoid_prime(z);
    }
};

class CrossEntropyCost : public Cost
{
    virtual float fn(const arma::fvec& a, const arma::fvec& y) override
    {
        arma::fvec result = -y * arma::log(a) - (1 - y) * arma::log(1 - a);
        result.replace(arma::datum::nan, 0.0f);
        // sum of all rows
        return arma::sum(result);
    }
    virtual arma::fmat error(const arma::fmat& z, const arma::fmat& a, const arma::fmat& y) override
    {
        return a - y;
    }
};
