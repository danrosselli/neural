import 'package:neural/neural.dart';
import 'package:console/console.dart';

void main() {
  // XOR training data
  var cursor = Cursor();

  // patterns of numbers from 0 to 9
  var trainingSet = [
    Learn(input: '111101101101111'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '001001001001001'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '111001111100111'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '111001111001111'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '101101111001001'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '111100111001111'.split('').map(double.parse).toList(), output: [1]), //5
    Learn(input: '111100111101111'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '111001001001001'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '111101111101111'.split('').map(double.parse).toList(), output: [0]),
    Learn(input: '111101111001111'.split('').map(double.parse).toList(), output: [0]),
  ];

  // Test data which contains distorted patterns of the number 5.
  final testData = [
    '111100111000111'.split('').map(double.parse).toList(),
    '111100010001111'.split('').map(double.parse).toList(),
    '111100011001111'.split('').map(double.parse).toList(),
    '110100111001111'.split('').map(double.parse).toList(),
    '110100111001011'.split('').map(double.parse).toList(),
    '111100101001111'.split('').map(double.parse).toList(),
  ];

  // Instantiation of neural network
  Network neuralNetwork = Network(input: 15, hidden: [3], output: 1);

  // Training of neural network
  print('\n');
  int oldEpoch = 0;
  neuralNetwork.train(trainingSet, 150000, 0.3, (epoch, input, output) {
    if (epoch != oldEpoch) {
      print('epoch: $epoch --------------------------------------------');
      cursor.moveUp(11);
    }
    print('Input: $input, Output: $output');
    oldEpoch = epoch;
  });

  // Test of rede neural
  print('\n');
  for (var data in trainingSet) {
    var inputs = data.input;
    var expectedOutput = data.output;
    var prediction = neuralNetwork.forward(inputs);
    print('Input: $inputs, Expected Output: $expectedOutput, Prediction: $prediction');
  }

  // Test generalization
  // First the number 5 itself.
  print('\n');
  final numberFive = trainingSet[5].input;
  print('Confidence in recognising a 5: ${neuralNetwork.forward(numberFive)}');

  // Distorted 5
  print('\n');
  for (var test in testData) {
    print('Confidence in recognising a distorted 5: ${neuralNetwork.forward(test)}');
  }

  print('\n');
  print('Is 0 = 5? ${neuralNetwork.forward(trainingSet[0].input)}');
  print('Is 8 = 5? ${neuralNetwork.forward(trainingSet[8].input)}');
  print('Is 3 = 5? ${neuralNetwork.forward(trainingSet[3].input)}');
}
