const { getMnistData } = require('./mnist-data-generation');
const { writeFileSync } = require('fs');
const math = require('mathjs');

const Network = layers => {

    const shuffle = array => {
        let m = array.length, t, i;
        while (m) {
            i = Math.floor(Math.random() * m--);
            t = array[m];
            array[m] = array[i];
            array[i] = t;
        }
    }



    const feedForward = inputs => {
        for (let layerIndex = 0; layerIndex < networkSize; ++layerIndex) { // layers
        }
    }


        }

        }
    }

        }
    }

    const predict = data => {
        const newData = data.map(data => {
            const output = feedForward(data.input).valueOf();
            const actual = output.indexOf(Math.max(...output));
            const ideal = data.output.indexOf(1);
            const cost = math.subtract(output, data.output).map(Math.abs);
            return {
                actual,
                ideal, 
                cost
            }
        }).reduce((acc, data) => ({
            accuracy: data.actual === data.ideal ? acc.accuracy + 1 : acc.accuracy, 
            cost: math.add(acc.cost, data.cost)
        }), { accuracy: 0, cost: math.zeros([data[0].output.length]) })

        newData.cost = math.divide(newData.cost, data.length);

        return newData;
    }

            }
        }
    }

    return {
        train, 
        predict
    }
}

(async () => {
    const data = await getMnistData();
    const epochs = 60;
    const miniBatchSize = 10;
    const learningRate = 0.1;
    const lambda = 5;
    const writeFrequency = 3000;
    const save = true;
    const showProgress = true
    const network = Network([28 * 28, 100, 10]);
    network.train(data, { epochs, miniBatchSize, learningRate, lambda }, { writeFrequency, save, showProgress });
    console.log('Final Accuracy: ', network.predict(data.testingData));
})()
