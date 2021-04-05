const { getMnistData } = require('./mnist-data-generation');
const { writeFileSync } = require('fs');

const math = require('mathjs');



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
            return {
                actual,
            }
    }

            }
        }
    }

    return {
        train, 
    }
}

    const network = Network([28 * 28, 100, 10]);
