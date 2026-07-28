export class PrintLifecycle {
  constructor({flush,cancelPagination,render,prepare,clearPresentation,print,restore,onAfterPrint,setFallback=(callback)=>setTimeout(callback,1000),clearFallback=clearTimeout}) {
    Object.assign(this,{flush,cancelPagination,render,prepare,clearPresentation,print,restore,onAfterPrint,setFallback,clearFallback});
    this.busy=false;
  }

  async run(){
    if(this.busy)return false;
    this.busy=true;
    const context=this.prepare();
    let fallback;
    let removeAfterPrint=()=>{};
    const cleanup=()=>{
      if(!this.busy)return;
      this.busy=false;
      if(fallback!==undefined)this.clearFallback(fallback);
      removeAfterPrint();
      this.restore(context);
    };
    try{
      this.flush();
      this.cancelPagination();
      if(!await this.render())throw new Error('The document could not be prepared for printing.');
      this.clearPresentation();
      removeAfterPrint=this.onAfterPrint(cleanup);
      this.print();
      if(this.busy)fallback=this.setFallback(cleanup);
      return true;
    }catch(error){
      cleanup();
      return false;
    }
  }
}
