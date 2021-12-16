
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
 * Stores nodes within a layer
 * Supported activation functions:
 *      -ReLU
 *      -Sigmoid
 * Supported error functions:
 *      -Mean Squared Error (MSE)
 *      -Cross Entropy
 */
function Layer(numNodes, activation="relu", error="mse") {
    // store the number of nodes for now, the weights will be determined when this layer is added to a neural network
    this.numNodes = numNodes;
    this.activation = activation;
    this.weights = null; // initialize the weights to be null as we don't know what dimension the weights should be until this layer is added to the network.
}

/**
 * Feeds the data to the nodes. Expects a column vector of the data matching the number of weights
 * Applys an activation function, if necessary.
 * @param {array} data 
 */
Layer.prototype.feed = function(data) {
    // begin by adding a bias row to the data
    let numEntries = math.subset(math.size(data), math.index(0))
    let bias = math.ones(numEntries, 1);
    data = math.concat(bias, data);
    // simply find the values at each node by taking the dot product of the weights of this layer
    // with the data
    let matrix = math.multiply(data, this.weights);
    // apply activation
    switch (this.activation) {
        case 'sigmoid':
            matrix = sigmoid(matrix);
            break;
    }
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
NeuralNetwork.prototype.addLayer = function(numNodes, activation="relu", error="mse") {
    let layer = new Layer(numNodes, activation, error);
    // determine the number of weights for the previous layer if there is one
    if (this.layers.length > 0) {
        // then initialize the weights of this layer
        let prevLayer = this.layers[this.layers.length - 1];
        // set the weights of that layer now (we only need weights for a layer that has an output layer)
        layer.weights = math.ones(prevLayer.numNodes + 1, layer.numNodes); // include an extra weight in each row for the bias
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

    // BACKWARD PHASE:
    let errors = [];
    for (var i = this.layers.length - 1; i >= 1; i--) {
        let layer = this.layers[i];
        let error;
        switch (layer.activation) {
            case 'sigmoid':
                // error = 
                break;
            case 'relu':
                break;
        }
    }

    console.log(outputs);
}

net = new NeuralNetwork();
net.addLayer(2);
net.addLayer(2);
net.addLayer(1, activation="sigmoid", error="crossentropy");

X = math.matrix(
    [[0, 0], [1, 1], [1, 0], [0, 1]]
);
// (a bias column is automatically attached to the data matrix when it is fed to the model)
Y = math.matrix(
    [[0], [0], [1], [1]]
);

net.backpropagate(X, Y);
