
/**
 * Stores nodes within a layer
 */
function Layer(numNodes, activation="relu") {
    // store the number of nodes for now, the weights will be determined when this layer is added to a neural network
    this.numNodes = numNodes;
    this.activation = activation;
    this.weights = null; // initialize the weights to be null as we don't know what dimension the weights should be until this layer is added to the network.
}

/**
 * Feeds the data to a node. Expects a column vector of the data matching the number of weights
 * @param {array} data 
 */
Layer.prototype.feed = function(data) {
    // simply find the values at each node by taking the dot product of the weights of this layer
    // with the data
    let matrix = math.multiply(this.weights, data);
    // apply activation
    switch (this.activation) {
        case 'sigmoid':
            matrix = math.multiply(matrix, -1);
            matrix = math.exp(matrix);
            matrix = math.add(matrix, 1);
            matrix = math.dotDivide(1, matrix);
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
NeuralNetwork.prototype.addLayer = function(numNodes, activation="relu") {
    let layer = new Layer(numNodes, activation);
    // determine the number of weights for the previous layer if there is one
    if (this.layers.length > 0) {
        // then initialize the weights of this layer
        let prevLayer = this.layers[this.layers.length - 1];
        // set the weights of that layer now (we only need weights for a layer that has an output layer)
        layer.weights = math.ones(layer.numNodes, prevLayer.numNodes); // include an extra weight in each row for the bias
    }
    // then add the layer
    this.layers.push(layer);
}

/**
 * Calculates the feed forward output of the given neural network
 */
NeuralNetwork.prototype.evaluate = function(data) {
    data = math.transpose(data); // transpose each entry into column vectors 
    for (var i = 1; i < this.layers.length; i++) {
        data = this.layers[i].feed(data);
    }
    return data;
}

net = new NeuralNetwork();
net.addLayer(2);
net.addLayer(2);
net.addLayer(1, activation="sigmoid");

console.log(net.layers);

X = math.matrix(
    [[0, 0], [1, 1], [1, 0], [0, 1]]
);
Y = math.matrix(
    [[0], [0], [1], [1]]
);

console.log(net.evaluate(X))