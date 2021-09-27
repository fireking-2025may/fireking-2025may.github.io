const initCanvas = ({ predict }) => {
    const predictButton = document.getElementById('predict');
    const deleteButton = document.getElementById('delete');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext("2d");

    const { width: canvasWidth, height: canvasHeight } = canvas;

    context.lineJoin = 'round';
    context.lineCap = 'round';
    context.lineWidth = 2.5;

    let isDrawing = false;
    let clearCanvas = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener('mousedown', e => {
        if (clearCanvas) {
            // context.fillStyle = '#fff';
            context.clearRect(0, 0, canvasWidth, canvasHeight)
            clearCanvas = false;
        };
        [lastX, lastY] = [e.offsetX * canvasWidth / 200, e.offsetY * canvasHeight / 200];
        isDrawing = true
    });

    canvas.addEventListener('mousemove', e => {
        if (!isDrawing) return;
        context.beginPath();
        context.moveTo(lastX, lastY);
        context.lineTo(e.offsetX * canvasWidth / 200, e.offsetY * canvasHeight / 200);
        context.stroke();
        context.closePath();
        [lastX, lastY] = [e.offsetX * canvasWidth / 200, e.offsetY * canvasHeight / 200];
    })

    canvas.addEventListener('mouseup', () => isDrawing = false);

    canvas.addEventListener('mouseout', () => isDrawing = false);

    predictButton.addEventListener('click', () => {
        const pixels = context.getImageData(0, 0, canvasWidth, canvasHeight);
        let inputs = [...Array(784)].fill(0);
        for (let pixelIndex = 3; pixelIndex < pixels.data.length; pixelIndex += 4) {
            const inputIndex = Math.floor((pixelIndex - 3) / (4));
            inputs[inputIndex] += pixels.data[pixelIndex];
        }
        console.log(inputs);
        document.getElementById('prediction').innerText = predict(inputs);
        clearCanvas = true;
    })

    deleteButton.addEventListener('click', () => context.clearRect(0, 0, canvasWidth, canvasHeight))

};

const Network = (coefficients) => {

    const sigmoid = value => 1 / (Math.exp(-value) + 1);

    const feedForward = inputs => {
        let activations = inputs;
        const networkSize = coefficients.length;
        for (let layerIndex = 0; layerIndex < networkSize; ++layerIndex) { // layers
            const layer = coefficients[layerIndex];
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

    const predict = inputs => {
        const outputs = feedForward(inputs);
        console.log(outputs.map(output => Math.round(output * 1000) / 1000));
        return outputs.indexOf(Math.max(...outputs));
    }

    return {
        feedForward, 
        predict
    }
}

(async () => (
    await (await fetch('/client/data/weights.json')).json()
))()
    .then(coefficients => initCanvas(Network(coefficients)))
