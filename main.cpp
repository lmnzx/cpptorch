#include <ATen/ops/randint.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <iomanip>
#include <iostream>
#include <torch/nn/functional/loss.h>
#include <torch/nn/modules/linear.h>
#include <torch/optim/sgd.h>
#include <torch/torch.h>

int main() {
  // setting the device to mps, as i am on a macbook
  torch::Device device(torch::kMPS);

  std::cout << "linear regression" << std::endl;

  // hyper parameters
  const int64_t input_size = 1;
  const int64_t output_size = 1;
  const size_t num_epochs = 60;
  const double learning_rate = 0.001;

  // sample dataset
  auto x_train = torch::randint(
      0, 10, {15, 1}, torch::TensorOptions(torch::kFloat).device(device));

  auto y_train = torch::randint(
      0, 10, {15, 1}, torch::TensorOptions(torch::kFloat).device(device));

  // linear regression model
  torch::nn::Linear model(input_size, output_size);
  model->to(device);

  // optimiser
  torch::optim::SGD optimizer(model->parameters(),
                              torch::optim::SGDOptions(learning_rate));

  std::cout << std::fixed << std::setprecision(4);

  std::cout << "Training...\n";

      // train the model
      for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
    // forward pass
    auto output = model->forward(x_train);
    auto loss = torch::nn::functional::mse_loss(output, y_train);

    // backwark pass and optimise
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 5 == 0) {
      std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                << "], Loss: " << loss.item<double>() << std::endl;
    }
  }

  std::cout << "training finished" << std::endl;

  return 0;
}
