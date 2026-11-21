(() => {
  const VOCAB=128, EMBED=96, HIDDEN=192, GATES=HIDDEN*4;
  const FLOATS=259712, BYTES=FLOATS*4;
  let weights=null, loading=null;
  async function load() {
    if(weights) return true;
    if(loading) return loading;
    loading=(async()=>{
      const response=await fetch('./models/honeydew_note_lm.f32');
      if(!response.ok) throw new Error('Не удалось загрузить файл модели: HTTP '+response.status);
      const buffer=await response.arrayBuffer();
      if(buffer.byteLength!==BYTES) throw new Error('Неверный размер модели: '+buffer.byteLength+' байт');
      const view=new DataView(buffer), values=new Float32Array(FLOATS);
      for(let i=0;i<FLOATS;i++) values[i]=view.getFloat32(i*4,true);
      for(let i=0;i<FLOATS;i++) if(!Number.isFinite(values[i])) throw new Error('В модели обнаружены некорректные веса');
      weights=values;
      return true;
    })().catch(err=>{loading=null;throw err;});
    return loading;
  }
  const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
  function nextLogits(pitch,hidden,cell) {
    let off=0;
    const embedding=off+pitch*EMBED; off+=VOCAB*EMBED;
    const wih=off; off+=GATES*EMBED;
    const whh=off; off+=GATES*HIDDEN;
    const bih=off; off+=GATES;
    const bhh=off; off+=GATES;
    const head=off; off+=VOCAB*HIDDEN;
    const headBias=off;
    const gates=new Float32Array(GATES);
    for(let g=0;g<GATES;g++){
      let v=weights[bih+g]+weights[bhh+g];
      let ir=wih+g*EMBED, hr=whh+g*HIDDEN;
      for(let i=0;i<EMBED;i++) v+=weights[ir+i]*weights[embedding+i];
      for(let i=0;i<HIDDEN;i++) v+=weights[hr+i]*hidden[i];
      gates[g]=v;
    }
    for(let i=0;i<HIDDEN;i++){
      const ig=sigmoid(gates[i]), fg=sigmoid(gates[HIDDEN+i]);
      const candidate=Math.tanh(gates[2*HIDDEN+i]), og=sigmoid(gates[3*HIDDEN+i]);
      cell[i]=fg*cell[i]+ig*candidate;
      hidden[i]=og*Math.tanh(cell[i]);
    }
    const logits=new Float32Array(VOCAB);
    for(let p=0;p<VOCAB;p++){
      let v=weights[headBias+p], row=head+p*HIDDEN;
      for(let i=0;i<HIDDEN;i++) v+=weights[row+i]*hidden[i];
      logits[p]=v;
    }
    return logits;
  }
  function generate(context,count,root,minor,seed) {
    if(!weights) throw new Error('Модель ещё не загружена');
    if(!context.length || context.some(p=>p<0||p>=VOCAB)) throw new Error('Некорректный музыкальный контекст');
    const scale=minor?[0,2,3,5,7,8,10]:[0,2,4,5,7,9,11];
    const allowed=[];
    for(let p=60;p<=95;p++) if(scale.includes(((p-60-(root%12)+12)%12))) allowed.push(p);
    if(!allowed.length) throw new Error('Не удалось определить тональность');
    const hidden=new Float32Array(HIDDEN), cell=new Float32Array(HIDDEN);
    let logits=new Float32Array(VOCAB);
    for(const pitch of context) logits=nextLogits(pitch,hidden,cell);
    let state=(seed>>>0)||1;
    const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
    const result=[];
    for(let n=0;n<count;n++){
      let max=-Infinity;
      for(const p of allowed) if(logits[p]>max) max=logits[p];
      const probs=[];let total=0;
      for(const p of allowed){const q=Math.exp((logits[p]-max)/0.85);probs.push(q);total+=q;}
      if(!Number.isFinite(total)||total<=0) throw new Error('Модель вернула некорректные вероятности');
      let pick=random()*total, selected=allowed[allowed.length-1];
      for(let i=0;i<allowed.length;i++){if(pick<=probs[i]){selected=allowed[i];break;}pick-=probs[i];}
      result.push(selected);logits=nextLogits(selected,hidden,cell);
    }
    return result;
  }
  window.HoneydewNoteLM={load,generate,get loaded(){return !!weights;}};
})();