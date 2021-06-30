const diagnostics = require('./data/diagnostics');
const fs = require('fs');

let diagnosticsCSV = 'Label, Training Accuracy, Training Cost, Testing Accuracy, Testing Cost, Validating Accuracy, Validating Cost\n';

diagnostics.forEach(({ epochIndex, trainingAcc, trainingCost, testingAcc, testingCost, validatingAcc, validatingCost }) => {
    diagnosticsCSV += `${epochIndex}, ${trainingAcc}, ${trainingCost}, ${testingAcc}, ${testingCost}, ${validatingAcc}, ${validatingCost}\n`;
})

fs.writeFileSync('./data/diagnostics.csv', diagnosticsCSV);