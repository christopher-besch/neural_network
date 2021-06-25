#pragma once
#include "hyper/data.h"
#include "hyper/hyper_surfer.h"
#include "learn/eval.h"
#include "learn/evaluator.h"
#include "learn/learn.h"
#include "main/log.h"
#include "net/costs.h"
#include "net/net.h"
#include "net/setup.h"

namespace NeuralNet {
inline void init() {
    Log::init();
}
} // namespace NeuralNet
