const math = require('mathjs');

const weights = [
    [ -2, -2 ],
    [ 2, 2 ]
]

const activation = [ 0, 1 ]

console.log(math.multiply(weights, activation));