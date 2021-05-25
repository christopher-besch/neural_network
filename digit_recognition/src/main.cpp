#include "pch.h"

#include "network.h"

#include <iostream>

int main()
{
    Network    net = Network({ 3, 3, 3 });
    arma::fvec a   = { 1, 2, 3 };
    net.feedforward(a);
    std::cout << a << std::endl;
    char s[5];
    s[10] = 5;
    return 0;
}
