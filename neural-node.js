var fs = require('fs');

const mnistDataGeneration = async () => {

    const network = require('./weights.json');

    let train = [];
    let test = [];
    
    const data = fs.readFileSync('mnist_train.csv', 'utf8').split('\n');

    data.pop();

    data.map(data => {
        data = data.split(',');
        const output = Array(10).fill(0);
        output[data.shift()] = 1
        const input = data.map(x => (+x) / 255);

        train.push({ output, input });
    })

    const testData = fs.readFileSync('mnist_test.csv', 'utf8').split('\n');

    testData.pop();

    testData.map(data => {
        data = data.split(',');
        const output = Array(10).fill(0);
        output[data.shift()] = 1
        const input = data.map(x => (+x) / 255);

        test.push({input, output});
    })

    return {
        train,
        test,
        network
    }
}

const Network = (layers, suppliedNetwork) => {

    const network = suppliedNetwork || layers.slice(1).map((layer, layerIndex) => { // layers
        return [...Array(layer)].map(() => { // neuron
            return {
                weights: [...Array(layers[layerIndex])].map(() => Math.random() * (1 - -1) - 1), // weights
                newWeights: [...Array(layers[layerIndex])].map(() => 0), // newWeights
                weightConstants: [...Array(layers[layerIndex])].map(() => ({ dirivative: 0, cost: 0 })),
                bias: Math.random() * (1 - -1) - 1,
                newBias: 0,
                output: 0, 
                delta: 1
            }
        })
    })

    const sigmoid = value => 1 / (Math.exp(-value) + 1);

    const sigmoidDirivative = output => output * (1 - output);

    const feedForward = inputs => {
        let activations = inputs;
        const networkSize = network.length;
        for (let layerIndex = 0; layerIndex < networkSize; ++layerIndex) { // layers
            const layer = network[layerIndex];
            const layerSize = layer.length;
            const newInputs = []
            for (let neuronIndex = 0; neuronIndex < layerSize; ++neuronIndex) { // neurons
                const neuron = layer[neuronIndex];
                let neuronTotal = 0;
                // let neuronTotal = neuron.bias;
                for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) { // weights
                    neuronTotal += neuron.weights[weightIndex] * activations[weightIndex];
                }
                const activation = sigmoid(neuronTotal);
                neuron.output = activation;
                newInputs.push(activation);
            }
            activations = newInputs;
        }
        return activations;
    }

    const backPropogation = (inputs, expectedOutput, learningRate) => {

        for (let layerIndex = network.length - 1; layerIndex >= 0; --layerIndex) {
            const layer = network[layerIndex];
            if (layerIndex !== network.length - 1) {
                for (let currentNeuronIndex = 0; currentNeuronIndex < layer.length; ++currentNeuronIndex) {
                    const neuron = layer[currentNeuronIndex];
                    let cost = 0;
                    for (let nextNeuronIndex = 0; nextNeuronIndex < network[layerIndex + 1].length; ++nextNeuronIndex) {
                        const nextNeuron = network[layerIndex + 1][nextNeuronIndex];
                        cost += nextNeuron.weightConstants[currentNeuronIndex].dirivative * nextNeuron.weightConstants[currentNeuronIndex].cost * nextNeuron.weights[currentNeuronIndex]
                    }
                    const dirivative = sigmoidDirivative(layer[currentNeuronIndex].output);
                    for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) {
                        let previousOutput;
                        if (layerIndex - 1 < 0) previousOutput = inputs[weightIndex]
                        else previousOutput = network[layerIndex - 1][weightIndex].output;
                        neuron.newWeights[weightIndex] -= (learningRate * previousOutput * dirivative * cost);
                        neuron.weightConstants[weightIndex].dirivative = dirivative;
                        neuron.weightConstants[weightIndex].cost = cost;
                    }
                    neuron.newBias -= (learningRate * dirivative * cost);
                }
            } else {
                for (let currentNeuronIndex = 0; currentNeuronIndex < layer.length; ++currentNeuronIndex) {
                    const neuron = layer[currentNeuronIndex];
                    const dirivative = sigmoidDirivative(neuron.output)
                    const cost = neuron.output - expectedOutput[currentNeuronIndex];
                    for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) {
                        const previousOutput = network[layerIndex - 1][weightIndex].output;
                        neuron.newWeights[weightIndex] -= (learningRate * previousOutput * dirivative * cost);
                        neuron.weightConstants[weightIndex].dirivative = dirivative;
                        neuron.weightConstants[weightIndex].cost = cost;
                    }
                    neuron.newBias -= (learningRate * dirivative * cost)
                }
            }
        }
    }

    const updateWeights = () => {
        for (let layerIndex = 0; layerIndex < network.length; ++layerIndex) { // layers
            for (let neuronIndex = 0; neuronIndex < network[layerIndex].length; ++neuronIndex) {
                const neuron = network[layerIndex][neuronIndex];
                for (let inputIndex = 0; inputIndex < neuron.weights.length; ++inputIndex) {
                    neuron.weights[inputIndex] = neuron.newWeights[inputIndex]
                }
                neuron.bias = neuron.newBias
            }
        }
    }

    const predict = data => {
        return data.map(data => {
            const output = feedForward(data.input);
            const actual = output.indexOf(Math.max(...output)) + 1;
            const ideal = data.output.indexOf(1) + 1;
            return {
                actual,
                ideal
            }
        }).reduce((acc, data) => data.actual === data.ideal ? acc +  1 : acc, 0) / data.length
    }

    const train = (data, epochs, batches, learningRate) => {
        for (let i = epochs; i > 0; --i) { // epoch
            for (let j = 0; j < batches + 2; ++j) { // batch
                const batch = data.splice(Math.floor(Math.random() * (data.length - (data.length / batches))), (data.length / batches));
                for (let dataIndex = 0; dataIndex < batch.length; ++dataIndex) {
                    const data = batch[dataIndex];
                    feedForward(data.input);
                    backPropogation(data.input, data.output, learningRate);
                    updateWeights(data.input);
                    console.log(`Epoch: ${i} | Batch: ${j}`);
                }
            }
        }
    }

    return {
        predict, 
        feedForward,
        train, 
        network
    }
}

const mnistData = mnistDataGeneration();

mnistData.then(value => {
    const epochs = 2;
    const batches = 1;
    const learningRate = 0.4;
    const network = Network([28 * 28, 14 * 14, 7 * 7, 10], value.network);
    network.train(value.train, epochs, batches, learningRate);
    console.log(network.predict(value.test));
    console.log(network);
})