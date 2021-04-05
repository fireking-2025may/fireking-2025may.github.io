const getMnistData = async () => {
    const trainingData = require('./data/mnist_train.json');
    const testingData = require('./data/mnist_test.json');
    const validatingData = require('./data/mnist_validate.json');

    return {
        trainingData, 
        testingData, 
        validatingData
    }
}

module.exports = {
    getMnistData
}