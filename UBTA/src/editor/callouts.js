export function bindCallouts(root=document){
  root.querySelectorAll('dialog').forEach(dialog=>{
    dialog.addEventListener('click',event=>{
      if(event.target===dialog)dialog.close('cancel');
    });
    dialog.querySelectorAll('[data-dialog-cancel]').forEach(button=>{
      button.addEventListener('click',event=>{
        event.preventDefault();
        dialog.close('cancel');
      });
    });
  });
}
