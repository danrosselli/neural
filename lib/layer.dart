import './neuron.dart';

class Layer {
  List<Neuron> neurons = [];

  // constructor, create a layer with n Neurons
  Layer(int totalNeurons) {
    for (int i = 0; i < totalNeurons; i++) {
      Neuron neuron = Neuron();
      neurons.add(neuron);
    }
  }

  List<double> forward(List<double> inputs) {
    // Iterates through the neurons within the layer, and for each neuron's output
    // adds it to the array, which will be used as input for the next layer
    List<double> result = [];
    for (var neuron in neurons) {
      result.add(neuron.forward(inputs));
      //print('--');
      //print('Neuron ${result.length} -> result ${result.last}');
    }
    return result;
  }

  double backward(double error, double learningRate) {
    for (var neuron in neurons) {
      error = neuron.backward(error, learningRate);
    }
    return error;
  }
}
