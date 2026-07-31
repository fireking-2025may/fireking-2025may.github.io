import Decimal from 'decimal.js';
// Thirty input digits leaves ample headroom for exact products and table sums.
// More importantly, it gives every syntactically valid input an explicit bound.
Decimal.set({precision:80,rounding:Decimal.ROUND_HALF_UP});
const DECIMAL=/^(?:0|[1-9]\d*)(?:\.\d{1,2})?$/;
const MAX_DIGITS=30;
export function parseDecimal(value) { if(!DECIMAL.test(value)||value.replace('.','').length>MAX_DIGITS) throw new Error(`Enter a non-negative number with no more than 2 decimal places and ${MAX_DIGITS} digits.`); return new Decimal(value); }
export function multiplyValues(a,b) { return parseDecimal(a).times(parseDecimal(b)).toDecimalPlaces(2,Decimal.ROUND_HALF_UP); }
export function formatShares(value) { return new Decimal(value).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g,','); }
export function formatMoney(value) { return `£${formatShares(value)}`; }
export function rowAggregate(row) { try{return formatMoney(multiplyValues(row.numberOfShares,row.nominalValuePerShare));}catch{return '—';} }
export function capitalTotals(rows) { let shares=new Decimal(0), nominal=new Decimal(0); for(const row of rows){try{const count=parseDecimal(row.numberOfShares), value=parseDecimal(row.nominalValuePerShare);shares=shares.plus(count);nominal=nominal.plus(count.times(value));}catch{ /* invalid rows do not contribute */ }} return {shares:formatShares(shares),nominal:formatMoney(nominal.toDecimalPlaces(2,Decimal.ROUND_HALF_UP))}; }
