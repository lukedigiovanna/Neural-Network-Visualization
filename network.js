
/**
 * Calculates an element wise sigmoid function for the given matrix.
 * @param {array} X Matrix to calculate element wise sigmoid function of
 * @param {boolean} derivative Whether or not to use the derivative of the sigmoid function
 */
function sigmoid(X, derivative=false) {
    if (derivative) {
        let sig = sigmoid(X);
        return math.dotMultiply(sig, math.subtract(1, sig))
    }
    else {
        return math.dotDivide(1, math.add(math.exp(math.multiply(X, -1)), 1));
    }
}

/**
 * Calculates an element wise identity function for the given matrix
 * @param {array} X Matrix to calculate element wise identity function of
 * @param {boolean} derivative Whether or not to use the derivative of the identity function
 * @returns 
 */
function identity(X, derivative=false) {
    if (derivative) {
        let size = math.size(X);
        return math.ones(math.subset(size, math.index(0)), math.subset(size, math.index(1)));
    }
    else {
        return X;
    }
}

/**
 * Calculates the mean squared error of a set of predictions and targets
 * @param {array} prediction Predicted values
 * @param {array} target Target values for predictions
 * @param {boolean} derivative Used to determine whether or not to find the derivative of this loss
 */
function meanSquaredError(prediction, target, derivative=false) {
    if (derivative) {
        return math.subtract(prediction, target);
    }
    else {
        let count = math.count(prediction);
        let err = math.square(math.subtract(prediction, target));
        // sum the individual errors
        let sum = 0;
        math.forEach(err, x => {
            sum += x;
        });
        return sum / count;
    }
}

/**
 * Calculates the cross-entropy error of a set of predictions and targets
 * Typically used for categorical predictors
 * @param {array} prediction Column vector of predicted values
 * @param {array} labels Column vector of labels
 * @param {boolean} derivative Used to determine whether or not to find the derivative of this loss
 */
function crossEntropy(prediction, labels, derivative=false) {
    if (derivative) {
        return math.add(math.dotDivide(math.dotMultiply(labels, -1), prediction), math.dotDivide(math.subtract(1, labels), math.subtract(1, prediction)));
    }
    else {
        let count = math.count(prediction);
        let err = math.add(math.dotMultiply(labels, math.log10(prediction)), math.dotMultiply(math.subtract(1, labels), math.log10(math.subtract(1, prediction))));
        let sum = 0;
        math.forEach(err, x => {
            sum += x;
        });
        return -sum / count;
    }
}

var activationFunctions = {
    "sigmoid": sigmoid,
    "identity": identity
};

var errorFunctions = new Map();
errorFunctions.set('mse', meanSquaredError);
errorFunctions.set('crossEntropy', crossEntropy);

/**
 * Adds a bias column to the leftmost column of the given matrix
 * @param {array} X Matrix to add bias column to
 */
function addBias(X) {
    let numEntries = math.subset(math.size(X), math.index(0))
    let bias = math.ones(numEntries, 1);
    return math.concat(bias, X);
}

/**
 * Stores nodes within a layer
 * Supported activation functions:
 *      -ReLU
 *      -Sigmoid
 * Supported error functions:
 *      -Mean Squared Error (MSE)
 *      -Cross Entropy
 */
function Layer(numNodes, activation="identity") {
    // store the number of nodes for now, the weights will be determined when this layer is added to a neural network
    this.numNodes = numNodes;
    this.activation = activationFunctions[activation];
    if (!this.activation) {
        throw "Unknown activation function: " + activation;
    }
    this.weights = null; // initialize the weights to be null as we don't know what dimension the weights should be until this layer is added to the network.
}

// Layer.prototype.activation = function(matrix, derivative=false) {
//     let func = activationFunctions.get(this.activationTitle);
//     console.log(func);
//     return func(matrix, derivative);
// }

/**
 * Feeds the data to the nodes. Expects a column vector of the data matching the number of weights
 * Applys an activation function, if necessary.
 * @param {array} data 
 */
Layer.prototype.feed = function(data) {
    // begin by adding a bias row to the data
    data = addBias(data);
    // simply find the values at each node by taking the dot product of the weights of this layer
    // with the data
    let matrix = this.activation(math.multiply(data, this.weights));
    return matrix;
}

/**
 * Stores a neural network
 */
function NeuralNetwork() {
    this.layers = [];
}

/**
 * Adds an initializes a layer to this network
 */
NeuralNetwork.prototype.addLayer = function(numNodes, activation="identity", error="mse") {
    let layer = new Layer(numNodes, activation, error);
    // determine the number of weights for the previous layer if there is one
    if (this.layers.length > 0) {
        // then initialize the weights of this layer
        let prevLayer = this.layers[this.layers.length - 1];
        // set the weights of that layer now (we only need weights for a layer that has an output layer)
        layer.weights = math.ones(prevLayer.numNodes + 1, layer.numNodes); // include an extra weight in each row for the bias
        layer.weights = math.map(layer.weights, x => {
            return x * (Math.random() * 2 - 1);
        });
    }
    // then add the layer
    this.layers.push(layer);
}

/**
 * Calculates the feed forward output of the given neural network
 */
NeuralNetwork.prototype.evaluate = function(data) {
    for (var i = 1; i < this.layers.length; i++) {
        data = this.layers[i].feed(data);
    }
    return data;
}

/**
 * Runs one iteration of the backpropagation algorithm on this neural network
 * @param {array} X The independent variable data to train the model on
 * @param {array} Y The target values
 * @param {number} learningRate The learning rate to use for gradient descent
 */
NeuralNetwork.prototype.backpropagate = function(X, Y, learningRate=0.01) {
    // FORWARD PHASE: 
    // store the output values for each layer
    let outputs = [X];
    for (var i = 1; i < this.layers.length; i++) {
        outputs.push(this.layers[i].feed(outputs[i - 1]));
    }
    for (var i = 0; i < this.layers.length - 1; i++) {
        outputs[i] = addBias(outputs[i]);
    }

    // BACKWARD PHASE:
    // error term = dC/da * g'(output)
    // calculate final layer error
    let activationDerivative = this.layers[this.layers.length - 1].activation(outputs[outputs.length - 1], derivative=true);
    let dCda = crossEntropy(outputs[outputs.length - 1], Y, derivative=true); // find the change in cross entropy less with respect to the outputs
    let outputError = math.dotMultiply(dCda, activationDerivative);
    let errors = [outputError];

    for (var i = this.layers.length - 2; i >= 1; i--) {
        let nextLayer = this.layers[i + 1];
        
        // find the activation derivative and the change in cost with respect to the subsequent error
        let activationDerivative = this.layers[i].activation(outputs[i], derivative=true); // use identity
        console.log(errors[0])
        let dCda = math.transpose(math.multiply(nextLayer.weights, math.transpose(errors[0])));
        
        // find the term error of the layer
        let error = math.dotMultiply(dCda, activationDerivative);
        console.log(error);
        return;
        errors.unshift(error);
    }
    
    // finally apply Gradient Descent
    // calculate the gradient for each layer
    let prevOutputs = addBias(outputs[1]);
    let numColumns = math.subset(math.size(prevOutputs), math.index(1));
    let numRows = math.subset(math.size(prevOutputs), math.index(0));
    let outputErrorFull = outputError;
    for (var i = 0; i < numColumns - 1; i++) {
        outputErrorFull = math.concat(outputErrorFull, outputError);
    }
    let outputPD = math.dotMultiply(prevOutputs, outputErrorFull);
    // average each column to get the gradient
    let gradient = [];
    for (var i = 0; i < numColumns; i++) {
        let column = math.subset(outputPD, math.index(math.range(0, numRows), i));
        // average the column
        let avg = 0;
        math.forEach(column, x => {
            avg += x;
        });
        avg /= numRows;
        gradient.push([avg]);
    }
    gradient = math.matrix(gradient);

    this.layers[2].weights = math.add(this.layers[2].weights, math.multiply(gradient, -learningRate));
    
    prevOutputs = addBias(outputs[0]);
    let hiddenPD = math.dotMultiply(prevOutputs, errors[0]);
    console.log(hiddenPD);
    gradient = [];
    for (var i = 0; i < numColumns; i++) {
        let column = math.subset(hiddenPD, math.index(math.range(0, numRows), i));
        // average the column
        let avg = 0;
        math.forEach(column, x => {
            avg += x;
        });
        avg /= numRows;
        gradient.push([avg]);
    }
    gradient = math.matrix(gradient);
    console.log(gradient);
    console.log(this.layers[1].weights);

    this.layers[1].weights = math.add(this.layers[1].weights, math.multiply(gradient, -learningRate));
    
    console.log(this.layers[2].weights);
}

net = new NeuralNetwork();
net.addLayer(2);
net.addLayer(2);
net.addLayer(1, activation="sigmoid", error="crossEntropy");

X = math.matrix(
    [[0, 0], [1, 1], [1, 0], [0, 1]]
);
// (a bias column is automatically attached to the data matrix when it is fed to the model)
Y = math.matrix(
    [[0], [0], [1], [1]]
);
// X = math.matrix(
//     [[0, 0]]
// );
// Y = math.matrix(
//     [[0]]
// );

for (var i = 0; i < 1; i++) {

    net.backpropagate(X, Y);
    // console.log(net.evaluate(X));
}

