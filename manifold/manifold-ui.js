(function () {
  'use strict';
  const M=window.ManifoldMath,$=id=>document.getElementById(id);
  const embedded=window.self!==window.top||new URLSearchParams(window.location.search).get('embedded')==='1';
  if(embedded)document.body.dataset.embedded='true';
  const C={bg:'#fbfaf7',ink:'#181a1b',muted:'#3d4348',rule:'#c4d0d5',blue:'#315f7f',orange:'#b45c2d',green:'#356c55',gray:'#667b86'};
  const defaults={curvature:.1,spacing:.8,sigmaMin:.2,iteration:0,probe:[.15,.045],showField:true,detail:false,detailCenter:[.15,.045],followRefinement:true};
  const starts=Array.from({length:15},(_,i)=>[-1.65+i*3.3/14+.013,.55+.19*Math.sin(i*1.7)+.12*Math.cos(i*.7)]);
  const bounds={xmin:-3.2,xmax:3.2,ymin:-.9,ymax:3.0};
  const canvases=['geometry-canvas','weights-canvas','outcomes-canvas'].map($);
  let state={...defaults,probe:defaults.probe.slice()},config,trajectories=[],field=[],pending=false,playing=false,maxIteration=1200,lastHeight=0,probeEvaluation,currentDiagnostics=[];
  let normalAxisRange=null,normalAxisTarget=.01,normalAxisBaseline=1,zoomAnimating=false;
  let previewMetrics=null,parameterComparison=null,parameterFeedback=null;
  let playbackTimer=null,zoomTimer=null,playRequested=false,playbackStart=0,playbackStartMass=0,playbackDuration=4800;
  let playbackMass=[0],visualEndIteration=0,probeCacheKey='',initialProjections=[],finalProjections=[];
  const frameInterval=33;
  function clearPlaybackTimer(){if(playbackTimer!==null){clearTimeout(playbackTimer);playbackTimer=null;}}
  function scheduleZoom(){if(!zoomAnimating||playing||zoomTimer!==null)return;zoomTimer=setTimeout(()=>{zoomTimer=null;drawOutcomes();scheduleZoom();},frameInterval);}
  function buildPlaybackMass(){
    // Time follows actual visible movement, rather than spending equal time
    // on hundreds of nearly stationary numerical refinement steps.
    playbackMass=[0];
    for(let n=1;n<=maxIteration;n++){
      let movement=0;
      for(const t of trajectories){const a=t.points[Math.min(n-1,t.points.length-1)],b=t.points[Math.min(n,t.points.length-1)];
        const world=Math.hypot(b.point[0]-a.point[0],b.point[1]-a.point[1])/state.spacing;
        const normal=Math.abs(b.normalError-a.normalError)/Math.max(.01*state.spacing,a.normalError);
        movement=Math.max(movement,world,.15*normal);}
      playbackMass.push(playbackMass[n-1]+Math.max(1e-8,movement));
    }
    const threshold=.995*playbackMass[maxIteration];visualEndIteration=playbackMass.findIndex(v=>v>=threshold);
    if(visualEndIteration<0)visualEndIteration=maxIteration;
  }
  function iterationAtMass(value){let lo=0,hi=maxIteration;while(lo<hi){const mid=(lo+hi+1)>>1;if(playbackMass[mid]<=value)lo=mid;else hi=mid-1;}return lo;}
  function startPlayback(){
    if(pending){playRequested=true;$('play-button').textContent='Computing…';return;}
    playRequested=false;clearPlaybackTimer();if(zoomTimer!==null){clearTimeout(zoomTimer);zoomTimer=null;}
    if(state.iteration>=maxIteration)state.iteration=0;
    playing=true;
    // A first real update is painted in the click handler, before any timer.
    setIteration(Math.min(maxIteration,state.iteration+1));
    if(state.iteration>=maxIteration){stop();return;}
    playbackStart=performance.now();playbackStartMass=playbackMass[state.iteration];
    playbackDuration=Math.min(4800,Math.max(1800,1800+70*Math.max(0,visualEndIteration-state.iteration)));
    playbackTimer=setTimeout(animate,frameInterval);
  }
  function animate(){
    playbackTimer=null;if(!playing)return;
    const progress=Math.min(1,Math.max(0,(performance.now()-playbackStart)/playbackDuration));
    const mass=playbackStartMass+progress*(playbackMass[maxIteration]-playbackStartMass);
    const next=progress>=1?maxIteration:Math.max(state.iteration,iterationAtMass(mass));
    if(next!==state.iteration)setIteration(next);else if(zoomAnimating)drawOutcomes();
    if(progress>=1){stop();return;}playbackTimer=setTimeout(animate,frameInterval);
  }
  function reportHeight(){if(!embedded)return;const height=Math.ceil(document.body.getBoundingClientRect().height+24);if(Math.abs(height-lastHeight)>1){lastHeight=height;parent.postMessage({type:'manifold-landscape-height',height},'*');}}
  function format(v,n=3){return Number.isFinite(v)?v.toFixed(n):'—';}
  function compact(v){if(!Number.isFinite(v))return '—';return Math.abs(v)<.001&&v!==0?v.toExponential(1):v.toFixed(3);}
  function sliderToSigma(v){return .03*Math.pow(1/.03,v/1000);}
  function sigmaToSlider(v){return Math.round(1000*Math.log(v/.03)/Math.log(1/.03));}
  function median(values){const a=values.filter(Number.isFinite).sort((a,b)=>a-b);if(!a.length)return NaN;const m=Math.floor(a.length/2);return a.length%2?a[m]:(a[m-1]+a[m])/2;}
  function precise(v){return Number.isFinite(v)?Number(v.toPrecision(3)).toString():'—';}
  function tolerance(){return 1e-6*state.spacing;}
  function updateTheoryRatios(){
    $('smoothing-ratio').textContent=precise(state.sigmaMin/state.spacing);
    $('curvature-ratio').textContent=precise(state.sigmaMin*state.sigmaMin*state.curvature/state.spacing);
  }
  function summarizePreview(){
    const end=trajectories.map(t=>infoAt(t,t.points.length-1)),counts={capture:0,between:0,away:0,refining:0},cells=new Set();
    end.forEach(d=>{counts[d.category]++;if(d.resolved&&!d.projection.tie)cells.add(d.cell);});
    return {counts,resolved:end.length-counts.refining,total:end.length,occupiedCells:cells.size,
      maximumResidual:Math.max(...end.map(d=>d.residual)),maximumRelativeResidual:Math.max(...end.map(d=>d.residual))/state.spacing,
      maximumUpdates:Math.max(...trajectories.map(t=>t.updates)),medianNormalError:median(end.map(d=>d.normalError)),
      medianTrainingDistance:median(end.map(d=>d.trainingDistance))};
  }
  function updateComputedFeedback(){
    if(!previewMetrics)return;
    const p=previewMetrics,c=p.counts;
    $('computed-preview').innerHTML='<strong>Computed numerical preview</strong> · '+p.resolved+'/'+p.total+' resolved after at most '+p.maximumUpdates+' updates: '+c.capture+' near training points; '+c.between+' between samples and near the curve; '+c.away+' away from the curve. '+p.occupiedCells+' resolved cells.'+(c.refining?' '+c.refining+' still refining; largest remaining pull / ℓ = '+precise(p.maximumRelativeResidual)+'.':'')+' This preview is independent of the playback step below.';
    $('parameter-feedback').hidden=!parameterFeedback;
    if(parameterFeedback){
      const f=parameterFeedback;
      $('parameter-feedback').textContent='At the same query, largest weight '+format(100*f.beforeTopShare,1)+'% → '+format(100*f.afterTopShare,1)+'%; effective contributors '+format(f.beforeContributors,2)+' → '+format(f.afterContributors,2)+'.'+(f.curvatureChanged?' Median normal distance of the computed outputs: '+precise(f.beforeMedianNormal)+' → '+precise(f.afterMedianNormal)+'.':'');
    }
  }
  function syncControls(){
    $('curvature').value=state.curvature;$('curvature-value').value=format(state.curvature,2);
    $('spacing').value=state.spacing;$('spacing-value').value=format(state.spacing,2);
    $('noise-floor').value=sigmaToSlider(state.sigmaMin);$('noise-floor-value').value=format(state.sigmaMin,2);
    $('iteration').max=maxIteration;$('iteration').value=state.iteration;$('iteration-value').value=state.iteration;$('iteration-max').textContent=maxIteration;
    $('show-field').checked=state.showField;$('play-button').textContent=playing?'Pause':playRequested?'Computing…':'Play';$('play-button').setAttribute('aria-pressed',String(playing));
    $('follow-refinement').checked=state.followRefinement;
    $('zoom-button').textContent=state.detail?'Show the full patch':'Zoom local detail';
    updateTheoryRatios();
  }
  function canvasContext(canvas){
    const box=canvas.getBoundingClientRect();if(box.width<80||box.height<80)return null;
    const dpr=Math.min(3,window.devicePixelRatio||1),w=box.width,h=box.height;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=C.bg;ctx.fillRect(0,0,w,h);return{ctx,w,h};
  }
  function line(ctx,points,color,width=1,dash){if(points.length<2)return;ctx.beginPath();ctx.moveTo(...points[0]);for(let i=1;i<points.length;i++)ctx.lineTo(...points[i]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash||[]);ctx.stroke();ctx.setLineDash([]);}
  function circle(ctx,p,r,color,stroke){if(!p.every(Number.isFinite)||r<0)return;ctx.beginPath();ctx.arc(p[0],p[1],r,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1.3;ctx.stroke();}}
  function diamond(ctx,p,r,color){ctx.beginPath();ctx.moveTo(p[0],p[1]-r);ctx.lineTo(p[0]+r,p[1]);ctx.lineTo(p[0],p[1]+r);ctx.lineTo(p[0]-r,p[1]);ctx.closePath();ctx.fillStyle=color;ctx.fill();}
  function square(ctx,p,r,color,outline=C.bg){ctx.fillStyle=color;ctx.fillRect(p[0]-r,p[1]-r,r*2,r*2);ctx.strokeStyle=outline;ctx.lineWidth=1.2;ctx.strokeRect(p[0]-r,p[1]-r,r*2,r*2);}
  function triangle(ctx,p,r,color,outline=C.bg,direction=1){ctx.beginPath();ctx.moveTo(p[0],p[1]-r*direction);ctx.lineTo(p[0]+r*.95,p[1]+r*.65*direction);ctx.lineTo(p[0]-r*.95,p[1]+r*.65*direction);ctx.closePath();ctx.fillStyle=color;ctx.fill();ctx.strokeStyle=outline;ctx.lineWidth=1.1;ctx.stroke();}
  function initialMarker(ctx,p,r=3.5){line(ctx,[[p[0]-r,p[1]-r],[p[0]+r,p[1]+r]],'#627c8b',1.5);line(ctx,[[p[0]-r,p[1]+r],[p[0]+r,p[1]-r]],'#627c8b',1.5);}
  function outcomeMarker(ctx,p,category,r=4.7){const color=categoryColor(category);if(category==='between')square(ctx,p,r,color);else if(category==='away')triangle(ctx,p,r+1,color);else if(category==='refining')circle(ctx,p,r,C.bg,color);else circle(ctx,p,r,color,C.bg);}
  function text(ctx,value,x,y,color=C.muted,size=14,align='left'){ctx.font=size+'px "Segoe UI",Arial,sans-serif';ctx.textAlign=align;ctx.fillStyle=color;ctx.fillText(value,x,y);}
  function pointLabels(ctx,items,T,obstacles){
    const occupied=[],layouts=[],padding=4;
    ctx.font='14px "Segoe UI",Arial,sans-serif';
    const intersects=(a,b)=>a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;
    for(const item of items){
      const width=Math.ceil(ctx.measureText(item.label).width)+8,height=21,other=items.find(v=>v!==item)?.point??item.point;
      const away=Math.atan2(item.point[1]-other[1],item.point[0]-other[0]);
      let best=null;
      for(const distance of[22,34,48,66])for(const offset of[0,Math.PI/4,-Math.PI/4,Math.PI/2,-Math.PI/2,Math.PI,3*Math.PI/4,-3*Math.PI/4]){
        const angle=away+offset,cx=item.point[0]+distance*Math.cos(angle),cy=item.point[1]+distance*Math.sin(angle);
        const box={x:Math.max(T.left+padding,Math.min(T.left+T.width-width-padding,cx-width/2)),y:Math.max(T.top+padding,Math.min(T.top+T.height-height-padding,cy-height/2)),w:width,h:height};
        const expanded={x:box.x-3,y:box.y-3,w:box.w+6,h:box.h+6};
        const markerHits=obstacles.reduce((n,p)=>n+(intersects(expanded,{x:p[0]-7,y:p[1]-7,w:14,h:14})?1:0),0);
        const labelHits=occupied.reduce((n,b)=>n+(intersects(expanded,b)?1:0),0);
        const cost=markerHits*10000+labelHits*100000+distance+Math.abs(offset)*2;
        if(!best||cost<best.cost)best={...box,cost,item};
      }
      occupied.push(best);layouts.push(best);
    }
    // Leaders and paper-colored callouts keep labels distinct without moving
    // the query, weighted average, or any of the computed training markers.
    for(const box of layouts){const p=box.item.point,c=[box.x+box.w/2,box.y+box.h/2],dx=c[0]-p[0],dy=c[1]-p[1],length=Math.hypot(dx,dy);
      if(length>12)line(ctx,[[p[0]+9*dx/length,p[1]+9*dy/length],c],'#7b8c95',1,[2,3]);
      ctx.fillStyle=C.bg;ctx.fillRect(box.x,box.y,box.w,box.h);text(ctx,box.item.label,box.x+4,box.y+15,box.item.color,14);
    }
  }
  function arrow(ctx,a,b,color,width=1){const len=Math.hypot(b[0]-a[0],b[1]-a[1]);if(len<.8)return;line(ctx,[a,b],color,width);const ux=(b[0]-a[0])/len,uy=(b[1]-a[1])/len,r=5;ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-r*ux+r*.45*uy,b[1]-r*uy-r*.45*ux);ctx.lineTo(b[0]-r*ux-r*.45*uy,b[1]-r*uy+r*.45*ux);ctx.closePath();ctx.fillStyle=color;ctx.fill();}
  function activeBounds(){if(!state.detail)return bounds;const x=state.detailCenter[0],y=.5*state.curvature*x*x;return{xmin:x-1.5*state.spacing,xmax:x+1.5*state.spacing,ymin:y-.65*state.spacing,ymax:y+.65*state.spacing};}
  function geometryTransform(w,h){
    const B=activeBounds(),roomW=w-60,roomH=h-53,scale=Math.min(roomW/(B.xmax-B.xmin),roomH/(B.ymax-B.ymin));
    const plotW=scale*(B.xmax-B.xmin),plotH=scale*(B.ymax-B.ymin),left=(w-plotW)/2,top=20;
    return{left,top,width:plotW,height:plotH,scale,xy:p=>[left+(p[0]-B.xmin)*scale,top+(B.ymax-p[1])*scale],world:p=>[B.xmin+(p[0]-left)/scale,B.ymax-(p[1]-top)/scale]};
  }
  function infoAt(t,n){
    const record=t.points[Math.min(n,t.points.length-1)],p=record.point,projection=M.project(p,config),resolved=record.residual<=tolerance()*1.001;
    let category='refining';if(resolved){if(record.trainingDistance<state.spacing/12)category='capture';else if(record.normalError<=state.spacing/10)category='between';else category='away';}
    return{...record,projection,category,resolved,cell:Math.round(projection.s/state.spacing),point:p};
  }
  function categoryColor(category){return category==='capture'?C.blue:category==='between'?C.green:category==='away'?C.orange:C.gray;}
  function drawGeometry(){
    const canvas=$('geometry-canvas'),width=canvas.getBoundingClientRect().width,B=activeBounds();
    if(width>=80){const desired=Math.max(205,Math.min(480,(width-60)*(B.ymax-B.ymin)/(B.xmax-B.xmin)+60));if(Math.abs(canvas.getBoundingClientRect().height-desired)>1)canvas.style.height=desired+'px';}
    const draw=canvasContext(canvas);if(!draw||!config)return;const{ctx,w,h}=draw,T=geometryTransform(w,h),xy=T.xy;
    ctx.save();ctx.beginPath();ctx.rect(T.left,T.top,T.width,T.height);ctx.clip();
    for(const x of [-3,-2,-1,0,1,2,3])line(ctx,[xy([x,B.ymin]),xy([x,B.ymax])],'#e7ebeb');
    for(const y of [0,1,2,3])line(ctx,[xy([B.xmin,y]),xy([B.xmax,y])],'#e7ebeb');
    const curve=[],upper=[],lower=[],tube=state.spacing/10;
    for(let i=0;i<=220;i++){const u=B.xmin+(B.xmax-B.xmin)*i/220,y=.5*state.curvature*u*u,norm=Math.hypot(1,state.curvature*u),nx=-state.curvature*u/norm,ny=1/norm;curve.push(xy([u,y]));upper.push(xy([u+tube*nx,y+tube*ny]));lower.push(xy([u-tube*nx,y-tube*ny]));}
    ctx.beginPath();ctx.moveTo(...upper[0]);upper.slice(1).forEach(p=>ctx.lineTo(...p));lower.reverse().forEach(p=>ctx.lineTo(...p));ctx.closePath();ctx.fillStyle='rgba(85,123,110,.08)';ctx.fill();
    if(state.showField)for(const f of field){const len=Math.hypot(...f.score);if(len>.0001){const p=xy(f.point),l=10;arrow(ctx,[p[0]-f.score[0]/len*l*.4,p[1]+f.score[1]/len*l*.4],[p[0]+f.score[0]/len*l*.6,p[1]-f.score[1]/len*l*.6],'rgba(49,95,127,.40)',1.1);}}
    line(ctx,curve,'#223c4d',2.4);
    for(const t of trajectories){const end=Math.min(state.iteration,t.points.length-1),stride=Math.max(1,Math.floor(end/100)),path=[];for(let i=0;i<=end;i+=stride)path.push(xy(t.points[i].point));if(end%stride)path.push(xy(t.points[end].point));line(ctx,path,'rgba(84,112,128,.48)',1.25);initialMarker(ctx,xy(t.start));}
    for(let j=0;j<config.centers.length;j++){
      const p=config.centers[j],q=probeEvaluation.q[j];if(q>.0015)line(ctx,[xy(state.probe),xy(p)],'rgba(180,92,45,'+Math.min(.90,.22+.68*q/probeEvaluation.topShare)+')',.9+2.4*q/probeEvaluation.topShare);
    }
    for(let j=0;j<config.centers.length;j++){const q=probeEvaluation.q[j];square(ctx,xy(config.centers[j]),q>.03?4.5:3.3,q>.03?C.blue:'#708da0',C.bg);}
    for(const diag of currentDiagnostics)outcomeMarker(ctx,xy(diag.point),diag.category,4.4);
    const p=xy(state.probe),bar=xy(probeEvaluation.barycenter),projection=M.project(state.probe,config);
    line(ctx,[p,xy(projection.point)],'#667d8a',1.4,[3,4]);arrow(ctx,p,bar,C.orange,2.6);diamond(ctx,bar,6.5,C.ink);circle(ctx,p,7.2,C.bg,C.orange);circle(ctx,p,2.7,C.orange);
    const labelObstacles=[p,bar,...config.centers.map(xy),...currentDiagnostics.map(d=>xy(d.point)),...trajectories.map(t=>xy(t.start))];
    const labels=[{label:'x',point:p,color:C.orange},{label:'μ̄',point:bar,color:C.ink}].filter(v=>v.point[0]>=T.left-.01&&v.point[0]<=T.left+T.width+.01&&v.point[1]>=T.top-.01&&v.point[1]<=T.top+T.height+.01);
    pointLabels(ctx,labels,T,labelObstacles);
    ctx.restore();ctx.strokeStyle=C.rule;ctx.lineWidth=1;ctx.strokeRect(T.left,T.top,T.width,T.height);
    const xTicks=state.detail?[B.xmin,(B.xmin+B.xmax)/2,B.xmax]:[-3,0,3],yTicks=state.detail?[B.ymin,(B.ymin+B.ymax)/2,B.ymax]:[0,1,2,3];
    for(const x of xTicks){const p=xy([x,B.ymin]);text(ctx,state.detail?format(x,1):String(x),p[0],p[1]+18,C.muted,14,'center');}
    for(const y of yTicks){const p=xy([B.xmin,y]);text(ctx,state.detail?format(y,1):String(y),p[0]-10,p[1]+4,C.muted,14,'right');}
    text(ctx,'x₁',T.left+T.width,T.top+T.height+35,C.muted,14,'right');text(ctx,'x₂',T.left+5,T.top-7,C.muted,14);
    text(ctx,'Reference curve',T.left+10,T.top+19,C.ink,14);
  }
  function drawWeights(){
    const draw=canvasContext($('weights-canvas'));if(!draw||!probeEvaluation)return;const{ctx,w,h}=draw,indices=probeEvaluation.q.map((q,j)=>({q,j})).sort((a,b)=>b.q-a.q).slice(0,8);
    const left=42,right=w-12,top=22,bottom=h-42,barW=(right-left)/indices.length,max=Math.max(.25,indices[0].q);
    for(const value of [0,max/2,max]){const y=bottom-(bottom-top)*value/max;line(ctx,[[left,y],[right,y]],'#e6eaea');text(ctx,Math.round(value*100)+'%',left-7,y+4,C.muted,14,'right');}
    indices.forEach((v,k)=>{const x=left+k*barW+barW*.14,bw=barW*.72,hh=(bottom-top)*v.q/max;ctx.fillStyle=k===0?C.orange:'#6388a1';ctx.fillRect(x,bottom-hh,bw,hh);text(ctx,String(v.j+1),x+bw/2,bottom+21,C.muted,14,'center');});
    text(ctx,'Training-point index',right,h-4,C.muted,14,'right');
    const shown=indices.reduce((a,v)=>a+v.q,0);$('weights-note').textContent='The largest 8 of 82 weights qⱼ in the local average; '+format(100*(1-shown),1)+'% remains outside these bars. The effective count is the equivalent number of equally weighted centers, using all 82.';
  }
  function drawOutcomes(){
    const draw=canvasContext($('outcomes-canvas'));if(!draw||!config)return;const{ctx,w,h}=draw;
    const initial=initialProjections,final=finalProjections;
    const coords=initial.concat(final).map(p=>p.s/state.spacing),lo=Math.floor(Math.min(...coords)-.7),hi=Math.ceil(Math.max(...coords)+.7);
    const currentExtent=Math.max(0,...currentDiagnostics.map(p=>Math.abs(p.projection.signedDistance/state.spacing))),needed=Math.max(.01,currentExtent*1.18);
    if(normalAxisRange===null){normalAxisRange=needed;normalAxisBaseline=needed;}
    normalAxisTarget=state.followRefinement?needed:Math.max(normalAxisRange,needed);
    // Expansion is immediate so no current output is hidden. Zooming inward
    // is eased in log-scale, preserving every plotted data coordinate.
    if(normalAxisTarget>=normalAxisRange)normalAxisRange=normalAxisTarget;
    else normalAxisRange=Math.exp(.78*Math.log(normalAxisRange)+.22*Math.log(normalAxisTarget));
    if(Math.abs(Math.log(normalAxisRange/normalAxisTarget))<.001)normalAxisRange=normalAxisTarget;
    zoomAnimating=Math.abs(Math.log(normalAxisRange/normalAxisTarget))>.001;scheduleZoom();
    const maxNormal=normalAxisRange,left=72,right=w-13,top=30,bottom=h-43,xy=(u,v)=>[left+(u-lo)/(hi-lo)*(right-left),bottom-(v+maxNormal)/(2*maxNormal)*(bottom-top)];
    ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();
    const bandA=xy(lo,.1),bandB=xy(hi,-.1);ctx.fillStyle='rgba(53,108,85,.10)';ctx.fillRect(bandA[0],bandA[1],bandB[0]-bandA[0],bandB[1]-bandA[1]);
    for(let k=lo;k<=hi;k++){if(hi-lo>14&&k%2)continue;const p=xy(k,0);line(ctx,[[p[0],top],[p[0],bottom]],'#d5dfe3');}
    line(ctx,[xy(lo,0),xy(hi,0)],'#567383',1.5);
    config.centerMeta.forEach(meta=>{const u=meta.s/state.spacing;if(u<lo||u>hi)return;square(ctx,xy(u,0),3.7,C.blue);});
    currentDiagnostics.forEach((diag,i)=>{const before=xy(initial[i].s/state.spacing,initial[i].signedDistance/state.spacing),after=xy(diag.projection.s/state.spacing,diag.projection.signedDistance/state.spacing);line(ctx,[before,after],'rgba(84,112,128,.38)',1.2);if(Math.abs(initial[i].signedDistance/state.spacing)<=maxNormal)initialMarker(ctx,before);});
    currentDiagnostics.forEach(diag=>outcomeMarker(ctx,xy(diag.projection.s/state.spacing,diag.projection.signedDistance/state.spacing),diag.category,5.1));
    ctx.restore();
    let hiddenStarts=0;initial.forEach(p=>{const v=p.signedDistance/state.spacing;if(Math.abs(v)<=maxNormal)return;hiddenStarts++;const x=xy(p.s/state.spacing,0)[0];triangle(ctx,[x,v>0?top+4:bottom-4],4,C.bg,'#627c8b',v>0?1:-1);});
    for(let k=lo;k<=hi;k++){if(hi-lo>14&&k%2)continue;const p=xy(k,0);text(ctx,String(k),p[0],bottom+22,C.muted,14,'center');}
    const axisFormat=v=>Math.abs(v)>=1?v.toFixed(1):Math.abs(v)>=.1?v.toFixed(2):v.toFixed(3);
    text(ctx,'0',left-10,xy(0,0)[1]+5,C.muted,14,'right');text(ctx,axisFormat(maxNormal),left-10,top+5,C.muted,14,'right');text(ctx,'−'+axisFormat(maxNormal),left-10,bottom+5,C.muted,14,'right');text(ctx,'s / ℓ',right,h-4,C.muted,14,'right');text(ctx,'normal / ℓ',left,16,C.muted,14);
    $('normal-range').textContent='±'+axisFormat(maxNormal)+'ℓ';$('normal-zoom').textContent=format(normalAxisBaseline/maxNormal,1)+'×';
    $('normal-zoom-note').textContent=(state.followRefinement?'The vertical axis follows the current outputs; their initial heights do not set its range. ':'Following is paused; the vertical axis holds its scale and expands only if needed. ')+(hiddenStarts?hiddenStarts+' initial positions are outside this zoom and indicated by gray edge triangles. ':'Gray crosses indicate the initial positions. ')+'Horizontal cell positions remain fixed. The pale band is normal distance ≤ 0.10ℓ.';
  }
  function updateReadouts(){
    const ev=probeEvaluation,projection=M.project(state.probe,config),training=M.nearestTraining(state.probe,config);
    $('top-share').textContent=format(100*ev.topShare,1)+'%';$('effective-count').textContent=format(ev.effectiveContributors,2);
    $('probe-reading').textContent=ev.topShare>.80?'One observed center dominates the weighted average. The bars show its coefficient, rather than a fraction of the final score magnitude.':ev.topShare>.5?'One neighbor leads, while other observations still alter the local average.':'Several observations share the local average. This can describe a neighborhood—and can also inherit curvature bias.';
    $('probe-location').textContent='At query x = ('+format(state.probe[0],2)+', '+format(state.probe[1],2)+'): training distance '+compact(training.distance/state.spacing)+'ℓ, manifold distance '+compact(projection.distance/state.spacing)+'ℓ.'+(projection.tie?' The nearest manifold projection is not unique here.':'');
    const counts={capture:0,between:0,away:0,refining:0},cells=new Set();currentDiagnostics.forEach(d=>{counts[d.category]++;if(d.resolved&&!d.projection.tie)cells.add(d.cell);});
    for(const key of Object.keys(counts))$(key+'-count').textContent=counts[key];
    const resolved=15-counts.refining;$('resolved-label').textContent=resolved+' / 15 numerically resolved';$('distinct-cells').textContent=resolved?cells.size:'—';
    $('training-distance').textContent=compact(median(currentDiagnostics.map(d=>d.trainingDistance))/state.spacing);$('manifold-distance').textContent=compact(median(currentDiagnostics.map(d=>d.normalError))/state.spacing);$('maximum-residual').textContent=compact(Math.max(...currentDiagnostics.map(d=>d.residual))/state.spacing);
    if(state.iteration===0)$('main-reading').innerHTML='Inspect the <strong>orange query point</strong> to see who determines the score. Then play the same fixed starts to see where this field takes them.';
    else if(counts.refining>0)$('main-reading').innerHTML='<strong>'+counts.refining+' trajectories are still refining.</strong> Nearness is reported separately from convergence; a finite iterate is not automatically an endpoint.';
    else if(counts.away>0)$('main-reading').innerHTML='<strong>'+counts.away+' outcomes are away from the curve</strong> at this resolution. Sharing many neighbors alone is insufficient: their average can be geometrically biased.';
    else if(counts.between>0)$('main-reading').innerHTML='<strong>'+counts.between+' outcomes lie between samples and near the curve.</strong> The '+cells.size+' resolved cells show how much variation this small ensemble retains.';
    else $('main-reading').innerHTML='The shown trajectories finish <strong>near individual training points</strong>. At this resolution the field behaves like a collection of sample-associated attractors.';
    $('model-details').textContent='Current settings: 82 observations; d = 3; σmin = '+format(state.sigmaMin,3)+'; σmax = 3; 32 scales; η = 0.35; refinement tolerance '+tolerance().toExponential(1)+'. The reference parabola has reach '+(Number.isFinite(config.reach)?format(config.reach,3):'∞')+'.';
    $('reach-value').textContent='Reach τ = '+(Number.isFinite(config.reach)?format(config.reach,2):'∞');
    updateComputedFeedback();
  }
  function render(){if(!config)return;syncControls();const key=state.probe.join(',');if(key!==probeCacheKey){probeEvaluation=M.evaluate(...state.probe,config);probeCacheKey=key;}currentDiagnostics=trajectories.map(t=>infoAt(t,state.iteration));drawGeometry();drawWeights();drawOutcomes();updateReadouts();setTimeout(reportHeight,0);}
  function rebuild(){
    const comparison=parameterComparison;
    config=M.prepare({curvature:state.curvature,spacing:state.spacing,sigmaMin:state.sigmaMin,sigmaMax:3,nScales:32,d:3});probeCacheKey='';
    trajectories=starts.map(start=>M.simulate(start,config,{eta:.35,maxSteps:1200,tolerance:tolerance()}));
    initialProjections=starts.map(p=>M.project(p,config));finalProjections=trajectories.map(t=>M.project(t.endpoint,config));
    previewMetrics=summarizePreview();parameterFeedback=null;parameterComparison=null;
    if(comparison&&comparison.probe.every((v,i)=>Math.abs(v-state.probe[i])<1e-12)&&['curvature','spacing','sigmaMin'].some(key=>Math.abs(comparison.parameters[key]-state[key])>1e-12)){
      const after=M.evaluate(...state.probe,config);
      parameterFeedback={beforeTopShare:comparison.topShare,afterTopShare:after.topShare,beforeContributors:comparison.effectiveContributors,
        afterContributors:after.effectiveContributors,beforeMedianNormal:comparison.medianNormalError,afterMedianNormal:previewMetrics.medianNormalError,
        curvatureChanged:Math.abs(comparison.parameters.curvature-state.curvature)>1e-12};
    }
    maxIteration=Math.max(1,...trajectories.map(t=>t.updates));state.iteration=Math.min(state.iteration,maxIteration);buildPlaybackMass();
    normalAxisRange=null;zoomAnimating=false;
    field=[];for(let x=-2.9;x<=2.91;x+=.58)for(let y=-.6;y<=2.71;y+=.55)field.push({point:[x,y],score:M.evaluate(x,y,config).score});
    render();
  }
  function stop(){playing=false;playRequested=false;clearPlaybackTimer();$('play-button').textContent='Play';$('play-button').setAttribute('aria-pressed','false');scheduleZoom();}
  function scheduleRebuild(){if(pending)return;pending=true;setTimeout(()=>{rebuild();pending=false;if(playRequested)startPlayback();},0);}
  function setParameters(patch){
    const changesModel=config&&['curvature','spacing','sigmaMin'].some(key=>key in patch&&Math.abs(+patch[key]-config[key])>1e-12);
    if(patch.probe){parameterComparison=null;parameterFeedback=null;}
    else if(changesModel&&!parameterComparison){
      const before=M.evaluate(...state.probe,config);
      parameterComparison={probe:state.probe.slice(),topShare:before.topShare,effectiveContributors:before.effectiveContributors,
        medianNormalError:previewMetrics?previewMetrics.medianNormalError:NaN,parameters:{curvature:config.curvature,spacing:config.spacing,sigmaMin:config.sigmaMin}};
    }
    stop();state={...state,...patch};state.curvature=Math.max(0,Math.min(.8,+state.curvature));state.spacing=Math.max(.35,Math.min(1.3,+state.spacing));state.sigmaMin=Math.max(.03,Math.min(1,+state.sigmaMin));state.iteration=0;if(patch.probe)state.probe=patch.probe.slice();
    if(changesModel){$('computed-preview').textContent='Computing this field from the same 15 fixed starts…';$('parameter-feedback').hidden=true;}
    syncControls();scheduleRebuild();
  }
  function setIteration(n){state.iteration=Math.max(0,Math.min(maxIteration,Math.round(n)));render();}
  function setProbe(point){if(!point||!point.every(Number.isFinite))return;parameterComparison=null;parameterFeedback=null;state.probe=[Math.max(bounds.xmin,Math.min(bounds.xmax,point[0])),Math.max(bounds.ymin,Math.min(bounds.ymax,point[1]))];render();}
  $('curvature').addEventListener('input',e=>setParameters({curvature:+e.target.value}));$('spacing').addEventListener('input',e=>setParameters({spacing:+e.target.value}));$('noise-floor').addEventListener('input',e=>setParameters({sigmaMin:sliderToSigma(+e.target.value)}));
  $('show-field').addEventListener('change',e=>{state.showField=e.target.checked;render();});
  $('follow-refinement').addEventListener('change',e=>{state.followRefinement=e.target.checked;render();});
  $('zoom-button').addEventListener('click',()=>{state.detail=!state.detail;state.detailCenter=state.probe.slice();render();});
  document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>setParameters({...M.presets[button.dataset.preset]})));
  $('play-button').addEventListener('click',()=>{if(playing||playRequested)stop();else startPlayback();});
  $('step-button').addEventListener('click',()=>{stop();setIteration(state.iteration>=maxIteration?0:state.iteration+1);});$('restart-button').addEventListener('click',()=>{stop();setIteration(0);});$('outcomes-button').addEventListener('click',()=>{stop();setIteration(maxIteration);});$('iteration').addEventListener('input',e=>{stop();setIteration(+e.target.value);});
  $('reset-probe').addEventListener('click',()=>setProbe(defaults.probe));$('reset-button').addEventListener('click',()=>{parameterComparison=null;parameterFeedback=null;state={...defaults,probe:defaults.probe.slice()};setParameters({});});
  let dragging=false;const geometry=$('geometry-canvas');
  function eventProbe(e){const rect=geometry.getBoundingClientRect(),T=geometryTransform(rect.width,rect.height);setProbe(T.world([e.clientX-rect.left,e.clientY-rect.top]));}
  geometry.addEventListener('pointerdown',e=>{dragging=true;geometry.setPointerCapture(e.pointerId);eventProbe(e);});geometry.addEventListener('pointermove',e=>{if(dragging)eventProbe(e);});geometry.addEventListener('pointerup',()=>dragging=false);geometry.addEventListener('pointercancel',()=>dragging=false);
  geometry.addEventListener('keydown',e=>{const directions={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,1],ArrowDown:[0,-1]};if(!directions[e.key])return;e.preventDefault();const delta=state.spacing*(e.shiftKey?.12:.04),dir=directions[e.key];setProbe([state.probe[0]+delta*dir[0],state.probe[1]+delta*dir[1]]);});
  let resizePending=false;new ResizeObserver(()=>{if(resizePending)return;resizePending=true;setTimeout(()=>{resizePending=false;render();},16);}).observe(document.body);
  new ResizeObserver(reportHeight).observe(document.body);
  window.manifoldDemo={getState:()=>({...state,probe:state.probe.slice(),pending,playing,maxIteration,playbackEndIteration:maxIteration,playbackVisualEndIteration:visualEndIteration,playbackDuration,playbackClock:'single monotonic timer',normalAxisRange,normalAxisTarget,normalZoomFactor:normalAxisRange?normalAxisBaseline/normalAxisRange:1,zoomAnimating,starts:starts.map(p=>p.slice()),theoryRatios:{smoothing:state.sigmaMin/state.spacing,geometricBias:state.sigmaMin*state.sigmaMin*state.curvature/state.spacing},previewMetrics:previewMetrics?{...previewMetrics,counts:{...previewMetrics.counts}}:null,parameterFeedback:parameterFeedback?{...parameterFeedback}:null,probeDiagnostics:probeEvaluation?{topShare:probeEvaluation.topShare,effectiveContributors:probeEvaluation.effectiveContributors,q:probeEvaluation.q.slice(),barycenter:probeEvaluation.barycenter.slice(),score:probeEvaluation.score.slice()}:null,currentDiagnostics:currentDiagnostics.map(d=>({...d,point:d.point.slice()})),trajectories:trajectories.map(t=>({start:t.start.slice(),endpoint:t.endpoint.slice(),converged:t.converged,residual:t.residual,updates:t.updates,minimumLogDensityIncrement:t.minimumLogDensityIncrement,points:t.points,diagnostics:t.diagnostics}))}),setParameters,setIteration,setProbe,reset:()=>{$('reset-button').click();}};
  pending=true;rebuild();pending=false;
})();
