/*! Paged.js compatibility adapter for the offline prototype.
 * Paged.js is MIT licensed: Copyright (c) 2018 Adam Hyde, Julie Blanc and contributors.
 * This constrained adapter implements the Previewer surface used here because the full upstream bundle was unavailable in the supplied materials. */
window.Paged={Previewer:class{async preview(content,styles,target){target.replaceChildren();const nodes=[...content.children];for(const node of nodes){const page=document.createElement('div');page.className='pagedjs_page';page.tabIndex=-1;const box=document.createElement('div');box.className='pagedjs_pagebox';box.append(node.cloneNode(true));page.append(box);target.append(page)}return{total:target.children.length,pages:[...target.children]}}}};
