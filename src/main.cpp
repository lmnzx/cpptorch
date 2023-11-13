#include <iostream>
#include <torch/torch.h>

#include "network.h"

int main() {
  Net network(50, 10);
  std::cout << network << std::endl;
  torch::Tensor x, output;
  x = torch::randn({2, 50});
  output = network->forward(x);
  std::cout << output << std::endl;
  return 0;
}
