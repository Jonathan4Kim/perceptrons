import perceptrons_data as data


############################################################
# Perceptrons
############################################################

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        # initialize weight vector
        self.w = {}
        # use iterations to iterate x amt of times told to
        for i in range(iterations):
            # for every x, y pair in examples
            for x, y in examples:
                # the weightedsum value, we will use for sign
                sign = 0
                # for every key value pair in x
                for key, val in x.items():
                    # if the key exists in the weights attribute
                    if key in self.w:
                        # add its weighted value to sign
                        sign += val * self.w[key]
                # during this iter, if our sign isn't consistent w y
                if (sign > 0) is not y:
                    # first, if y is True
                    if y:
                        # iterate through all of the keys
                        for key in x.keys():
                            # if the key is already in weights
                            if key in self.w:
                                # add its value to self.w value
                                self.w[key] += x[key]
                            else:
                                # otherwise, create it
                                self.w[key] = x[key]
                    # otherwise, y is false, so we work neg.
                    else:
                        # iterate through all of x's keys
                        for key in x.keys():
                            # check if it exists in the dictionary
                            if key in self.w:
                                # subtract its value from the val in self.w
                                self.w[key] -= x[key]
                            else:
                                # otherwise, create it in self.w
                                self.w[key] = -x[key]

    def predict(self, x):
        # weighted sum value of 0
        sum = 0
        # for every key in x.keys
        for key in x.keys():
            # get the weight of the key stored in self.w
            weight = self.w.get(key)
            # if it is not None
            if weight:
                # add its weighted value to the sum
                sum += x[key] * weight
        # return the boolean of if the weighted sum is positive
        return sum > 0


class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        # create empty nested dictionary for every y val in examples
        self.lw = {y: {} for (x, y) in examples}
        # iterate the given amount of times
        for i in range(iterations):
            # iterated through all pairs in examples
            for x, actualY in examples:
                # set predicted to None, initialization
                pred = None
                # maximum to a negative value to start
                maximum = -1
                # compute predicted labels as argmax here
                for y in self.lw.keys():
                    # weighted sum of this y is 0 to start
                    sum = 0
                    # get the weighted map attribute we've store
                    wmap = self.lw.get(y)
                    # iterate through every key in x
                    for key in x.keys():
                        # get the weight of the key in the dict
                        weight = wmap.get(key)
                        # if the weight isn't none
                        if weight:
                            # then we add its weighted value to sum
                            sum += x[key] * weight
                    # if the sum is the max, we need a new argmax
                    if sum > maximum:
                        # set new maximum value
                        maximum = sum
                        # set new argmax
                        pred = y
                # Similar to last time, we check if our predicted value
                if pred != actualY:
                    # we get the correct mapping by using actualY
                    correct_label_map = self.lw.get(actualY)
                    # for every key in x
                    for key in x.keys():
                        # we check if it's in the correct map
                        if key in correct_label_map:
                            # and if so, we add the val of x to it, updating it
                            correct_label_map[key] += x[key]
                        # otherwise it's not in the map
                        else:
                            # and we thus get an actual positive initialization
                            correct_label_map[key] = x[key]
                    # then we get our predicted map
                    pred_map = self.lw.get(pred)
                    # for every key in x.keys
                    for key in x.keys():
                        # we check if it's in the predicted map
                        if key in pred_map:
                            # and we take away our x value from it
                            pred_map[key] -= x[key]
                        # otherwise, we need to initialize the value
                        else:
                            # so we do so here, but in a negative fashion
                            pred_map[key] = -x[key]

    def predict(self, x):
        # prediction, initialized to None
        pred = None
        # a nonpositive maximum value, will store max value
        maximum = -1
        # for every y in the dictionary
        for y in self.lw.keys():
            # the sum, initialized to 0
            sum = 0
            # get the weighted map from the dictionary
            wmap = self.lw.get(y)
            # iterate through every key in x
            for key in x.keys():
                # get the weighted value
                weight = wmap.get(key)
                # if the weight isn't none
                if weight:
                    # add it to the sum
                    sum += x[key] * weight
            # if at any point, the sum is the maximum
            if sum > maximum:
                # store the sum value as the new maximum
                maximum = sum
                # the prediction is y
                pred = y
        # return the predicted value
        return pred

############################################################
# Applications
############################################################


class IrisClassifier(object):

    def __init__(self, data):
        # call MutliClassPerceptron with 8 iterations, reading the data
        self.classifier = MulticlassPerceptron(get_data(data), 100)

    def classify(self, instance):
        # call the predict function, formatting the instance properly
        return self.classifier.predict(format(instance))


class DigitClassifier(object):

    def __init__(self, data):
        # call MutliClassPerceptron with 8 iterations, getting the data
        self.classifier = MulticlassPerceptron(get_data(data), 8)

    def classify(self, instance):
        # call the predict function, formatting the instance properly
        return self.classifier.predict(format(instance))


class BiasClassifier(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron([({1: x, 2: 1}, y)
                                            for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1: instance, 2: 1})


class MysteryClassifier1(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron([({1: x[0] ** 2 +
                                              x[1] ** 2, 2: 1}, y)
                                            for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1:
                                        instance[0] ** 2 + instance[1] ** 2,
                                        2: 1})


class MysteryClassifier2(object):

    def __init__(self, data):
        self.classifier = BinaryPerceptron([({1: x[0] * x[1] * x[2]}, y)
                                            for x, y in data], 10)

    def classify(self, instance):
        return self.classifier.predict({1:
                                        instance[0] *
                                        instance[1] * instance[2]})


def get_data(data):
    return [({i: v for i, v in enumerate(x, 1)}, y) for x, y in data]


def format(instance):
    return {i + 1: instance[i] for i, x in enumerate(instance, 0)}

############################################################
# ain
############################################################


def main():
    train = ([({"x1": 1}, True), ({"x2": 1}, True),
              ({"x1": -1}, False), ({"x2": -1}, False)])
    test = [{"x1": 1}, {"x1": 1, "x2": 1},
            {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]
    p = BinaryPerceptron(train, 1)
    print([p.predict(x) for x in test])
    # MultiClassPerceptron
    train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3),
             ({"x1": -1, "x2": 1}, 4),
             ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),
             ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]
    # Train the classifier for 10 iterations so that it can learn each class
    p = MulticlassPerceptron(train, 10)
    print([p.predict(x) for x, y in train])
    c = IrisClassifier(data.iris)
    print(c.classify((5.1, 3.5, 1.4, 0.2)))
    c = IrisClassifier(data.iris)
    print(c.classify((7.0, 3.2, 4.7, 1.4)))
    c = DigitClassifier(data.digits)
    print(c.classify((0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13,
                      15, 10, 15, 5, 0, 0, 3,
                      15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0, 0, 5, 8, 0,
                      0, 9, 8, 0, 0, 4,
                      11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10,
                      12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0)))
    c = BiasClassifier(data.bias)
    print([c.classify(x) for x in (-1, 0, 0.5, 1.5, 2)])
    c = MysteryClassifier1(data.mystery1)
    print([c.classify(x) for x in
           ((0, 0), (0, 1), (-1, 0), (1, 2), (-3, -4))])
    c = MysteryClassifier2(data.mystery2)
    print([c.classify(x) for x in ((1, 1, 1),
                                   (-1, -1, -1), (1, 2, -3), (-1, -2, 3))])


if __name__ == "__main__":
    main()