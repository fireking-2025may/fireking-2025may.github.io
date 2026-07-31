/** @typedef {import('../zip/index.js').EnclosureFiles} EnclosureFiles */
/** @typedef {{caseData:import('../data/model.js').S77Case,files:EnclosureFiles}} ImportedCase */
/** @param {import('../data/model.js').S77Case} _caseData @param {EnclosureFiles} _files */
export async function exportCaseBundle(_caseData,_files){throw new Error('Not implemented in this step');}
/** @param {Uint8Array} _bytes @returns {Promise<ImportedCase>} */
export async function importCaseBundle(_bytes){throw new Error('Not implemented in this step');}
