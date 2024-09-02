# Perceptron Classifiers

## Overview

This project implements both binary and multiclass perceptron classifiers. The perceptron algorithm is a supervised learning technique used for binary linear classification. In the multiclass case, it can be extended to handle multiple classes.

## Classifiers

### BinaryPerceptron

The `BinaryPerceptron` class is designed to handle binary classification problems. It takes as input a list of examples and a specified number of iterations for training.

- **Initialization:**  
  The perceptron initializes a weight vector and iteratively updates it based on the training data. The weights are adjusted when the predicted class does not match the actual class.

- **Prediction:**  
  The `predict` method computes the weighted sum of the input features and returns `True` if the sum is positive, indicating class `1`, or `False` otherwise.

### MulticlassPerceptron

The `MulticlassPerceptron` class extends the perceptron algorithm to handle multiple classes. It maintains a weight vector for each class and updates the weights based on the prediction accuracy.

- **Initialization:**  
  Similar to the binary perceptron, the multiclass perceptron iteratively updates the weights for each class over a specified number of iterations.

- **Prediction:**  
  The `predict` method calculates the weighted sum for each class and returns the class with the highest sum, implementing an `argmax` function.

## Applications

Several specific classifiers are implemented using the perceptron classes:

### IrisClassifier

- **Purpose:**  
  Classifies the famous Iris dataset into different species using the `MulticlassPerceptron`.

### DigitClassifier

- **Purpose:**  
  Classifies handwritten digits, leveraging the `MulticlassPerceptron` to differentiate between numerical classes.

### BiasClassifier

- **Purpose:**  
  Demonstrates the binary perceptron's behavior when trained on biased data, illustrating the perceptron's ability to handle skewed distributions.

### MysteryClassifier1 & MysteryClassifier2

- **Purpose:**  
  Two mystery classifiers showcase the perceptron's adaptability to different feature transformations and combinations.

## Usage

To run the classifiers, ensure you have the necessary data imported from `homework9_data`. You can execute the `main()` function to see the classifiers in action with predefined datasets.

### Example

```python
train = [({"x1": 1}, True), ({"x2": 1}, True), ({"x1": -1}, False), ({"x2": -1}, False)]
test = [{"x1": 1}, {"x1": 1, "x2": 1}, {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]
p = BinaryPerceptron(train, 1)
print([p.predict(x) for x in test])
```

## Reflection/Feedback

### What I Found Challenging

The most challenging aspect of this project was implementing and understanding the differences between various classifiers and ensuring each one was properly adjusted to handle its specific type of input and classification logic.

### What I Enjoyed

I enjoyed exploring the differences between binary and multiclass perceptrons, particularly how the algorithm can be adapted to handle more complex classification problems.
