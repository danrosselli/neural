import './layer.dart';
import 'types.dart';

class Network {
  int input;
  List<int> hidden;
  int output;

  List<Layer> layers = [];

  bool verbose = false;

  Network({required this.input, required this.hidden, required this.output}) {
    createLayers();
  }

  createLayers() {
    // don't need create a layer for input layer, it is the pure input number
    //Layer inputLayer = Layer(input);
    //layers.add(inputLayer);

    // create a layer with n neurons for hidden layer
    for (int n in hidden) {
      Layer hiddenLayer = Layer(n);
      layers.add(hiddenLayer);
    }

    // create a layer with n neurons for output layer
    Layer outputLayer = Layer(output);
    layers.add(outputLayer);
  }

  List<double> forward(List<double> inputs) {
    var current = inputs;
    //int count = 0;
    for (var layer in layers) {
      //print('----------------------------------------------------');
      //print('Layer $count');
      current = layer.forward(current);
      //count++;
    }
    //print('----------------------------------------------------');
    return current;
  }

  void train(List<Learn> trainingData, int epochs,
      [double learningRate = 0.1, void Function(int epoch, List<double>, List<double>)? onIterate]) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      for (var data in trainingData) {
        // get training data
        var input = data.input;
        var expectedOutput = data.output;

        // Forward pass
        var output = forward(input);

        // if has function to show this round results
        if (onIterate != null) {
          onIterate(epoch, input, output);
        }

        // Backward pass (Backpropagation)
        backward(expectedOutput, learningRate);
      }
    }
  }

  void backward(List<double> expectedOutput, [double learningRate = 0.1]) {
    var error = calculateError(expectedOutput);

    for (var i = layers.length - 1; i >= 0; i--) {
      error = layers[i].backward(error, learningRate);
    }
  }

  double calculateError(List<double> expectedOutput) {
    var outputLayer = layers.last;
    var output = outputLayer.neurons.map((neuron) => neuron.output).toList();
    var errors = List<double>.generate(output.length, (index) {
      return expectedOutput[index] - output[index];
    });
    return errors.fold(0, (sum, element) => sum + element);
  }
}
