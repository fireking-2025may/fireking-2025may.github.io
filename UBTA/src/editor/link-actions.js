import { safeHref } from '../state/schema.js';

export function followEditorLink(href,{navigateInternal,openExternal,openMailto,report=()=>{}}){
  const safe=safeHref(href);
  if(!safe){report('This link is not safe to open');return false}
  if(safe.startsWith('#')){const opened=navigateInternal(safe.slice(1));if(!opened)report('The linked document destination no longer exists');return opened}
  if(/^https?:/i.test(safe)){openExternal(safe);return true}
  if(/^mailto:/i.test(safe)){openMailto(safe);return true}
  return false;
}
