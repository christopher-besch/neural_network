#pragma once
#include "costs.h"
#include "net/hyper_surfer.h"
#include "net/learn.h"
#include "net/net.h"
#include "net/setup.h"
#include "read_mnist.h"

namespace NeuralNet
{
inline void init()
{
    Log::init();
}
} // namespace NeuralNet
