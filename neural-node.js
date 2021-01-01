var fs = require('fs');

const mnistDataGeneration = async () => {

    const data = fs.readFileSync('mnist_train.csv', 'utf8').split('\n');

    data.pop();

    const train = data.map(data => {
        data = data.split(',');
        const output = Array(10).fill(0);
        output[data.shift()] = 1
        const input = data.map(x => (+x) / 255);
        return {
            input,
            output
        };
    })

    const testData = fs.readFileSync('mnist_test.csv', 'utf8').split('\n');
    testData.pop();
    const test = testData.map(data => {
        data = data.split(',');
        const output = Array(10).fill(0);
        output[data.shift()] = 1
        const input = data.map(x => (+x) / 255);
        return {
            input,
            output
        };
    })

    return {
        train,
        test
    }
}

const Network = (layers, suppliedNetwork) => {

    const network = suppliedNetwork || layers.slice(1).map((layer, layerIndex) => [...Array(layer)].map(() => ({ // layers // neurons
            weights: [...Array(layers[layerIndex])].map(() => Math.random() * (1 - -1) - 1), // weights
            newWeights: [...Array(layers[layerIndex])].map(() => 0), // newWeights
            weightConstants: [...Array(layers[layerIndex])].map(() => ({ dirivative: 0, cost: 0 })),
            bias: Math.random() * (1 - -1) - 1,
            newBias: 0,
            output: 0, 
            delta: 1
        }))
    )

    const shuffle = array => {
        let m = array.length, t, i;
        while (m) {
            i = Math.floor(Math.random() * m--);
            t = array[m];
            array[m] = array[i];
            array[i] = t;
        }
    }

    const getRandomSubArray = (array, newLength) => {
        const newArray = new Array(newLength);
        let length = array.length;
        const taken = new Array(length);
        while (newLength--) {
            const i = Math.floor(Math.random() * length);
            newArray[i] = array[i in taken ? taken[i] : i];
            taken[i] = --length in taken ? taken[length] : length;
        }
        return newArray;
    }

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
                let neuronTotal = neuron.bias;
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

    const singleBackPropogation = (inputs, expectedOutput, learningRate) => {
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

    const backPropogation = (inputs, expectedOutput, learningRate) => {
        for (let layerIndex = network.length - 1; layerIndex >= 0; --layerIndex) {
            const layer = network[layerIndex];
            const layerLength = layer.length;
            if (layerIndex !== network.length - 1) {
                for (let currentNeuronIndex = 0; currentNeuronIndex < layerLength; ++currentNeuronIndex) {
                    const neuron = layer[currentNeuronIndex];
                    let cost = 0;
                    for (let nextNeuronIndex = 0; nextNeuronIndex < network[layerIndex + 1].length; ++nextNeuronIndex) {
                        const nextNeuron = network[layerIndex + 1][nextNeuronIndex];
                        const nextNeuronWeightConstants = nextNeuron.weightConstants[currentNeuronIndex]
                        cost += nextNeuronWeightConstants.dirivative * nextNeuronWeightConstants.cost * nextNeuron.weights[currentNeuronIndex]
                    }
                    const dirivative = sigmoidDirivative(neuron.output);
                    for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) {
                        let previousOutput;
                        if (layerIndex - 1 < 0) previousOutput = inputs[weightIndex]
                        else previousOutput = network[layerIndex - 1][weightIndex].output;
                        neuron.newWeights[weightIndex] += (neuron.newWeights[weightIndex] - (learningRate * previousOutput * dirivative * cost));
                        neuron.weightConstants[weightIndex].dirivative = dirivative;
                        neuron.weightConstants[weightIndex].cost = cost;
                    }
                    neuron.newBias += (neuron.newBias - (learningRate * dirivative * cost));
                }
            } else {
                for (let currentNeuronIndex = 0; currentNeuronIndex < layer.length; ++currentNeuronIndex) {
                    const neuron = layer[currentNeuronIndex];
                    const dirivative = sigmoidDirivative(neuron.output)
                    const cost = neuron.output - expectedOutput[currentNeuronIndex];
                    for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) {
                        const previousOutput = network[layerIndex - 1][weightIndex].output;
                        neuron.newWeights[weightIndex] += neuron.newWeights[weightIndex] - (learningRate * previousOutput * dirivative * cost);
                        neuron.weightConstants[weightIndex].dirivative = dirivative;
                        neuron.weightConstants[weightIndex].cost = cost;
                    }
                    neuron.newBias += (neuron.newBias - (learningRate * dirivative * cost));
                }
            }
        }
    }

    const updateSingleWeights = () => {
        for (let layerIndex = 0; layerIndex < network.length; ++layerIndex) { // layers
            for (let neuronIndex = 0; neuronIndex < network[layerIndex].length; ++neuronIndex) { // neurons
                const neuron = network[layerIndex][neuronIndex];
                for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) { // weights
                    neuron.weights[weightIndex] = neuron.newWeights[weightIndex];
                }
                neuron.bias = neuron.newBias;
            }
        }
    }

    const updateWeights = batchLength => {
        for (let layerIndex = 0; layerIndex < network.length; ++layerIndex) { // layers
            for (let neuronIndex = 0; neuronIndex < network[layerIndex].length; ++neuronIndex) { // neurons
                const neuron = network[layerIndex][neuronIndex];
                for (let weightIndex = 0; weightIndex < neuron.weights.length; ++weightIndex) { // weights
                    neuron.weights[weightIndex] = neuron.newWeights[weightIndex] / batchLength
                    neuron.newWeights[weightIndex] = 0;
                }
                neuron.bias = neuron.newBias / batchLength
                neuron.newBias = 0;
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

    const train = (data, epochs, batchLength, learningRate, testData) => {
        const batches = data.length / batchLength
        for (let i = 0; i <= epochs; ++i) { // epochs
            shuffle(data);
            for (let j = 0; j < batches; ++j) { // batches
                const randomIndex = Math.floor(Math.random() * (data.length - batchLength - 2));
                const batch = data.slice(randomIndex, randomIndex + batchLength);
                for (let dataIndex = 0; dataIndex < batchLength; ++dataIndex) {
                    const data = batch[dataIndex];
                    feedForward(data.input);
                    singleBackPropogation(data.input, data.output, learningRate);
                    updateSingleWeights();
                }
                console.log(`Epoch: ${i} | Batch: ${j} | Accuracy: ${predict(testData)}`);
            }
        }
        fs.writeFileSync('data/weights.json', JSON.stringify(network));
    }

    return {
        predict, 
        feedForward,
        train, 
        network
    }
}

mnistDataGeneration().then(value => {
    const epochs = 2;
    const batchLength = 10;
    const learningRate = 0.5;
    const network = Network([28 * 28, 100, 10]);
    network.train(value.train, epochs, batchLength, learningRate, value.test);
    console.log('Final Accuracy: ', network.predict(value.test));
})