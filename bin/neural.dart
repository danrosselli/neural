import 'package:neural/neural.dart';
import 'package:neural/trainer.dart';
import 'package:console/console.dart';

void main() {
  // XOR training data
  var cursor = Cursor();

  var trainingSet = [
    Learn(input: [0, 0], output: [0]),
    Learn(input: [1, 0], output: [1]),
    Learn(input: [0, 1], output: [1]),
    Learn(input: [1, 1], output: [0]),
  ];

  // Instanciação da rede neural
  Network neuralNetwork = Network(input: 2, hidden: [3], output: 1);

  // Treinamento da rede neural
  int oldEpoch = 0;
  neuralNetwork.train(trainingSet, 100000, 0.3, (epoch, input, output) {
    if (epoch != oldEpoch) {
      print('-- $epoch --------------------------------------------');
      cursor.moveUp(5);
    }
    print('Input: $input, Output: $output');
    oldEpoch = epoch;
  });

  // Teste da rede neural
  for (var data in trainingSet) {
    var inputs = data.input;
    var expectedOutput = data.output;
    var prediction = neuralNetwork.forward(inputs);
    print('Input: $inputs, Expected Output: $expectedOutput, Prediction: $prediction');
  }
}
