export class History{
  constructor(value){this.past=[];this.future=[];this.value=value;this.gesture=null}
  commit(value){this.past.push(structuredClone(this.value));this.value=value;this.future=[];return this.value}
  beginGesture(){if(!this.gesture)this.gesture=structuredClone(this.value);return this.value}
  updateGesture(value){if(!this.gesture)this.beginGesture();this.value=value;return this.value}
  endGesture(value=this.value){if(!this.gesture)return this.value;const before=this.gesture;this.gesture=null;this.value=value;if(JSON.stringify(before)!==JSON.stringify(value)){this.past.push(before);this.future=[]}return this.value}
  cancelGesture(){if(this.gesture){this.value=this.gesture;this.gesture=null}return this.value}
  undo(){if(this.gesture)this.cancelGesture();if(!this.past.length)return this.value;this.future.push(structuredClone(this.value));return this.value=this.past.pop()}
  redo(){if(!this.future.length)return this.value;this.past.push(structuredClone(this.value));return this.value=this.future.pop()}
}
