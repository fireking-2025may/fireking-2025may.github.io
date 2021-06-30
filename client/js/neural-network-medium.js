const { writeFileSync } = require('fs');

const math = require('mathjs');

const Network = layers => {
    let weights = [...Array(layers.length - 1)].map((_, i) => math.random([layers[i + 1], layers[i]], -1 / math.sqrt(layers[i + 1]), 1 / math.sqrt(layers[i + 1])));
    let biases = [...Array(layers.length - 1)].map(((_, i) => math.random([layers[i + 1]], -1, 1)));

    const shuffle = array => {
        let m = array.length, t, i;
        while (m) {
            i = Math.floor(Math.random() * m--);
            t = array[m];
            array[m] = array[i];
            array[i] = t;
        }
        return array;
    }

    const sigmoid = z => math.dotDivide(1, math.add(1, math.exp(math.multiply(-1, z))))

    const sigmoidDerivative = activation => math.dotMultiply(activation, math.subtract(1, activation));

    const costDerivative = (outputs, ideal) => math.subtract(math.flatten(outputs), ideal);

    const feedForward = input => {
        const networkSize = layers.length - 1;
        for (let layerIndex = 0; layerIndex < networkSize; ++layerIndex) { // layers
            input = sigmoid(math.add(math.multiply(weights[layerIndex], input), biases[layerIndex]));
        }
        return input;
    }

    const backPropagation = ({input, ideal}) => {

        let deltaWeights = [...Array(layers.length - 1)].map((_, i) => math.zeros([layers[i + 1], layers[i]]));
        let deltaBiases = [...Array(layers.length - 1)].map(((_, i) => math.zeros([layers[i + 1]])));

        let activation = input;
        let activations = [input];
        let zs = [];

        const networkSize = layers.length - 1;
        for (let layerIndex = 0; layerIndex < networkSize; ++layerIndex) { // layers
            const z = math.add(math.multiply(weights[layerIndex], activation), biases[layerIndex]);
            zs.push(z);
            activation = sigmoid(z);
            let newActivation = math.matrix(activation);
            math.reshape(newActivation, [newActivation.size()[0], 1]);
            activations.push(newActivation);
        }

        let deltaVector = costDerivative(activations[activations.length - 1], ideal);
        deltaVector = math.reshape(deltaVector, [deltaVector.size()[0], 1])
        deltaBiases[deltaBiases.length - 1] = deltaVector
        deltaWeights[deltaWeights.length - 1] = math.multiply(deltaVector, math.transpose(activations[activations.length - 2]));

        for (let layerIndex = deltaBiases.length - 2; layerIndex >= 0; --layerIndex) {
            deltaVector = math.dotMultiply(math.multiply(math.transpose(weights[layerIndex + 1]), deltaVector), sigmoidDerivative(activations[layerIndex + 1]))
            deltaVector = math.reshape(deltaVector, [deltaVector.size()[0], 1])
            deltaBiases[layerIndex] = deltaVector;
            deltaWeights[layerIndex] = math.multiply(deltaVector, math.transpose(math.reshape(math.matrix(activations[layerIndex]), [activations[layerIndex].length, 1])));
        }

        return { deltaWeights, deltaBiases };
    }

    const updateMiniBatch = (miniBatch, learningRate, lambda, trainingDataLength) => {
        let newWeights = [...Array(layers.length - 1)].map((_, i) => math.zeros([layers[i + 1], layers[i]]));
        let newBiases = [...Array(layers.length - 1)].map(((_, i) => math.zeros([layers[i + 1]])));
        for (let dataIndex = 0; dataIndex < miniBatch.length; ++dataIndex) {
            const data = miniBatch[dataIndex];
            const { deltaWeights, deltaBiases } = backPropagation(data);
            newWeights = [...Array(layers.length - 1)].map((_, i) => math.add(newWeights[i], deltaWeights[i]));
            newBiases = [...Array(layers.length - 1)].map((_, i) => math.add(newBiases[i], math.flatten(deltaBiases[i])));
        }

        weights = [...Array(layers.length - 1)].map((_, i) => math.subtract(math.dotMultiply(weights[i], 1 - (learningRate * lambda / trainingDataLength)), math.dotMultiply(learningRate / miniBatch.length, newWeights[i])));
        biases = [...Array(layers.length - 1)].map((_, i) =>  math.subtract(biases[i], math.dotMultiply(learningRate / miniBatch.length, newBiases[i])));
    }

    const predict = data => {
        return data.map(data => {
            const output = feedForward(data.input).valueOf();
            return {
                actual: output.indexOf(Math.max(...output)) + 1,
                ideal: data.ideal.indexOf(1) + 1
            }
        }).reduce((acc, data) => data.actual === data.ideal ? acc +  1 : acc, 0) / data.length
    }

    const train = ({ trainingData, testingData, validatingData }, { epochs, miniBatchSize, learningRate, lambda, save }) => {
        console.log('Training');
        const trainingDataLength = trainingData.length;
        for (let epochIndex = 0; epochIndex < epochs; ++epochIndex) {
            trainingData = shuffle(trainingData);
            for (let batchIndex = 0; batchIndex < trainingDataLength; batchIndex += miniBatchSize) {
                if(!(batchIndex % 1)) console.log(`Batch index ${batchIndex}`);
                const miniBatch = trainingData.slice(batchIndex, batchIndex + miniBatchSize);
                updateMiniBatch(miniBatch, learningRate, lambda, trainingDataLength);
            }
            if (testingData) {
                console.log(`Epoch ${epochIndex} ${predict(testingData) * testingData.length} / ${testingData.length}`);
            } else {
                console.log(`Epoch ${epochIndex}`);
            }
            if(save) writeFileSync('client/data/weights.json', JSON.stringify({ weights, biases }));
        }
    }

    return {
        train, 
        predict
    }
}



const network = Network([3, 2, 2]);

const trainingData = [{input: [1, 0, 1], ideal: [1, 0]}];
const testingData = [{input: [1, 0, 1], ideal: [1, 0]}];

const epochs = 10;
const miniBatchSize = 1;
const learningRate = 1;
const lambda = 1;
const save = false;

network.train({ trainingData, testingData }, { epochs, miniBatchSize, learningRate, lambda, save })
console.log(network.predict(testingData));