/** @typedef {'England'|'Scotland'|'Wales'|'Northern Ireland'} Jurisdiction */
/** @typedef {'A'|'B'|'C'|'D'|'E'|'F'|'G'|'H'|'I'} EnclosureCode */
/** @typedef {{name:string,companyNumber:string,registeredOffice:string,incorporationDate:string,jurisdiction:Jurisdiction|string}} Company */
/** @typedef {{id:string,shareClass:string,numberOfShares:string,nominalValuePerShare:string,shareRights:string,paymentStatus:string}} CapitalRow */
/** @typedef {{id:string,heading:string,body:string}} CommercialReason */
/** @typedef {{id:string,numberOfShares:string,shareClass:string,nominalValuePerShare:string}} ScheduleShareLine */
/** @typedef {{id:string,sellerName:string,saleShares:ScheduleShareLine[],considerationShares:ScheduleShareLine[]}} ScheduleSeller */
/** @typedef {{category:EnclosureCode,order:number,originalFilename:string,generatedFilename:string,mimeType:string,size:number,sha256:string}} EnclosureManifestEntry */
/** @typedef {{schemaVersion:1,appVersion:string,exportedAt?:string,form:{ourReference:string,transactionDate:string,acquiringCompany:Company,targetCompany:Company,acquiringPreTransactionCapital:CapitalRow[],targetPreTransactionCapital:CapitalRow[],acquiringPostTransactionCapital:CapitalRow[],commercialReasons:CommercialReason[],agreement:{saleShareColumnNumber:string,considerationShareColumnNumber:string,scheduleReference:string},schedule:ScheduleSeller[],adviserName:string},enclosures:Record<EnclosureCode,EnclosureManifestEntry[]>,documentsArchive?:{filename:'Documents Referenced - S77.zip',size:number,sha256:string}}} S77Case */

export const APP_VERSION = '1.0.0';
let counter = 0;
export const newId = () => `item-${Date.now().toString(36)}-${++counter}`;
export const blankCompany = () => ({name:'',companyNumber:'',registeredOffice:'',incorporationDate:'',jurisdiction:''});
export const blankCapitalRow = (paymentStatus='Fully Paid') => ({id:newId(),shareClass:'',numberOfShares:'',nominalValuePerShare:'',shareRights:'',paymentStatus});
export const blankShareLine = () => ({id:newId(),numberOfShares:'',shareClass:'',nominalValuePerShare:''});
export const blankSeller = () => ({id:newId(),sellerName:'',saleShares:[blankShareLine()],considerationShares:[blankShareLine()]});
/** @returns {S77Case} */
export function createBlankCase() {
  return {schemaVersion:1,appVersion:APP_VERSION,form:{ourReference:'',transactionDate:'',adviserName:'',acquiringCompany:blankCompany(),targetCompany:blankCompany(),acquiringPreTransactionCapital:[blankCapitalRow()],targetPreTransactionCapital:[blankCapitalRow()],acquiringPostTransactionCapital:[blankCapitalRow()],commercialReasons:[{id:newId(),heading:'',body:''}],agreement:{saleShareColumnNumber:'',considerationShareColumnNumber:'',scheduleReference:''},schedule:[blankSeller()]},enclosures:{A:[],B:[],C:[],D:[],E:[],F:[],G:[],H:[],I:[]}};
}
