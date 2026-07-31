export function isoToDisplay(iso) { if(!/^\d{4}-\d{2}-\d{2}$/.test(iso)) return ''; const [y,m,d]=iso.split('-'); return `${d}/${m}/${y}`; }
export function currentDateDisplay(date=new Date()) { return new Intl.DateTimeFormat('en-GB',{day:'2-digit',month:'2-digit',year:'numeric'}).format(date); }
