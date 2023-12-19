import 'dart:math';

class Neuron {
  late List<double> inputs;
  List<double> weights = [];

  double bias = 1;
  late double output;
  late double gradient;

  double forward(List<double> inputs) {
    this.inputs = inputs;

    // fill the weights with random values in the first use
    if (weights.isEmpty) {
      for (int i = 0; i < this.inputs.length; i++) {
        weights.add(generateRandomNumber(0.3, 0.7));
      }
      //weights = List.filled(this.inputs.length, 0.5);
    }

    double sum = 0;
    for (int i = 0; i < inputs.length; i++) {
      sum += inputs[i] * weights[i];
    }
    sum += bias;

    // activation function
    output = sigmoid(sum);
    //output = relu(sum);
    //print('inputs: $inputs | weights: $weights | bias: $bias | sum: $sum | out: $output');

    return output;
  }

  double backward(double error, double learningRate) {
    // Calculate the gradient (derivative) of the sigmoid activation function
    gradient = output * (1 - output);

    // Calculate the gradient (derivative) of the ReLU activation function
    //gradient = output > 0 ? 1 : 0;

    // Update weights and bias based on the error, the gradient, and the learning rate
    for (int i = 0; i < weights.length; i++) {
      weights[i] += (learningRate * 10) * error * gradient * inputs[i];
    }
    bias += (learningRate * 10) * error * gradient;

    // Return the error to be passed to the previous layer
    return error * gradient * weights.reduce((sum, weight) => sum + weight);
  }

  double generateRandomNumber(double min, double max) {
    // Criando uma instância de Random
    Random random = Random();

    // Gerando um número aleatório entre 0.0 (inclusive) e 1.0 (exclusive)
    double randomDouble = random.nextDouble();

    // Mapeando o número aleatório para o intervalo desejado
    double result = min + (randomDouble * (max - min));

    return result;
  }

  // activation sgmoid (always between 0 and 1)
  double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  // activation ReLU (Rectified Linear Unit)
  double relu(double x) {
    // original approach
    //return max(0, x);
    // approach to normalize ReLU between 0 and 1
    return max(0, x) / (1 + exp(-x));
  }
}
