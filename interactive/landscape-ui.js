(function () {
  'use strict';
  const M = window.GaussianLandscapeMath;
  const embedded = window.self !== window.top || new URLSearchParams(window.location.search).get('embedded') === '1';
  if (embedded) document.body.dataset.embedded = 'true';
  let lastPostedHeight = 0;
  function reportHeight() {
    if (!embedded) return;
    const height = Math.ceil(document.body.getBoundingClientRect().height + 24);
    if (Math.abs(height-lastPostedHeight)>1) {lastPostedHeight=height;window.parent.postMessage({type:'gaussian-landscape-height',height},'*');}
  }
  const $ = id => document.getElementById(id);
  const BLUE = '#416e90', ORANGE = '#c77849', INK = '#181a1b', MUTED = '#3d4348', BG = '#fbfaf7';
  const COLORS = [BLUE, '#8295a0', ORANGE];
  const MANY = [[-2.2,-.3],[-.9,1.6],[1.3,1.35],[2,-1.15],[-.45,-1.65]];
  const LIMIT = 4.2, MAX_ITER = 120, N_SCALES = 32;
  const defaults = {stage:0,d:16,sigmaMin:.32,sigmaMax:1.35,eta:.03,rule:'raw',height:'log',target:'range',iteration:0};
  let state = {...defaults}, config, surface, contour, farField, trajectories = [], extraStarts = [], playing = false;
  let yaw = .72, pitch = .46, pending = false, resizeTimer = null;
  let playbackTimer = null, playbackStarted = 0, playbackFrom = 0, playbackDuration = 0;
  let playbackEndIteration = MAX_ITER, playbackIterations = [0], playbackFrames = [0], gridRevision = 0, surfaceCache = null, contourCache = null;
  let gridConfigKey = '', farFieldConfigKey = '';
  const POSITION_TOLERANCE = 2e-4;
  const canvases = ['shell-canvas','profile-canvas','surface-canvas','contour-canvas','far-field-canvas'].map($);
  const contexts = new Map();
  const stories = [
    ['Equal draws occupy very different volumes','At each noise scale, draw the same number of examples. In high dimensions they lie near radius σ√d. Shell volume scales as σᵈ, while effective mixture mass scales as σ². For d > 2, volume grows faster.'],
    ['The scales combine into one mountain','Around one center, combine 32 Gaussian components with σ² weights. The broad components extend the foothills; the narrow components concentrate density near the summit.'],
    ['Nearby mountains shape one another','Add five centers without changing the rule. Their contributions overlap into a fixed landscape. The score points toward a state-dependent average of the centers.'],
    ['One fixed field, many climbing paths','Keep the terrain fixed and move each point along its score. Adjust η to see gradual ascent, oscillation, or overshoot. The same starting points are reused whenever a parameter changes.']
  ];
  function rng(seed) { let x=seed>>>0; return ()=>{x=(Math.imul(1664525,x)+1013904223)>>>0;return (x+.5)/4294967296;}; }
  function normal(random) { return Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random()); }
  const random = rng(1739);
  const starts = Array.from({length:18},()=>[Math.max(-3.7,Math.min(3.7,normal(random)*2)),Math.max(-3.7,Math.min(3.7,normal(random)*2))]);
  const shellRandom = rng(287), normals = Array.from({length:72},()=>Array.from({length:128},()=>normal(shellRandom))), angles = Array.from({length:72},()=>2*Math.PI*shellRandom());
  const baseStarts = () => starts.concat(extraStarts).map(p=>p.slice());
  function isSingle() { return state.stage===1 || (state.stage===3 && state.target==='single'); }
  function chooseConfig() { return M.prepare({d:state.d,sigmaMin:state.sigmaMin,sigmaMax:state.sigmaMax,nScales:N_SCALES,centers:isSingle()?[[0,0]]:MANY}); }
  function etaRange() { return state.rule==='raw'?[.002,.45]:[.02,2.5]; }
  function etaToSlider(eta) { const [a,b]=etaRange();return Math.round(1000*Math.log(eta/a)/Math.log(b/a)); }
  function sliderToEta(t) { const [a,b]=etaRange();return a*Math.pow(b/a,t/1000); }
  function format(v,n=2) { return Number.isFinite(v)?v.toFixed(n):'—'; }
  function syncControls() {
    $('dimension').value=state.d;$('dimension-value').value=state.d;
    $('sigma-min').value=state.sigmaMin;$('sigma-min-value').value=format(state.sigmaMin);
    $('sigma-max').value=state.sigmaMax;$('sigma-max-value').value=format(state.sigmaMax);
    $('step-size').value=etaToSlider(state.eta);$('step-size-value').value=format(state.eta,3);
    const [a,b]=etaRange();$('eta-min').textContent=format(a,3);$('eta-max').textContent=format(b,3);
    $('gradient-rule').value=state.rule;$('ascent-centers').value=state.target;
    $('iteration').max=playbackEndIteration;$('iteration').value=state.iteration;$('iteration-value').value=state.iteration;$('iteration-end').textContent=playbackEndIteration;
    const active=state.stage===3;
    $('play-button').disabled=pending||!active;
    $('step-size').disabled=!active;document.querySelector('.step-parameter').classList.toggle('inactive',!active);
    document.querySelector('.height-toggle').classList.toggle('inactive',state.stage===0);
    document.querySelectorAll('[data-stage]').forEach(el=>{const active=+el.dataset.stage===state.stage;el.classList.toggle('selected',active);el.setAttribute('aria-pressed',active);});
    document.querySelectorAll('[data-height]').forEach(el=>{const active=el.dataset.height===state.height;el.classList.toggle('selected',active);el.setAttribute('aria-pressed',active);});
    $('shell-scene').hidden=state.stage!==0;$('landscape-scene').hidden=state.stage===0;$('playback').hidden=!active;$('slice-note').hidden=state.stage===0;$('far-field-panel').hidden=state.stage===0;
    $('stage-heading').textContent=stories[state.stage][0];$('stage-description').textContent=stories[state.stage][1];
    $('surface-title').textContent=isSingle()?'One fixed density':'The fixed density landscape';
    $('height-label').textContent=state.height==='density'?'Height: p / max shown':'Full '+(surface?format(surface.range,1):'')+'-unit log range';
    $('surface-note').textContent=state.height==='density'?'Drag to rotate. Density height is p divided by the largest value shown. Narrow peaks can hide changes in the broad tails.':'Drag to rotate. Height rescales the full displayed log-density range to [0,1], retaining the broad foothills.';
    $('contour-title').textContent=active?'The climb from above':'The uphill directions';
    $('contour-hint').textContent=active?'Click to add a start':'Score arrows';
    $('contour-note').textContent='Contours cover the full displayed log-density range'+(surface?' ('+format(surface.range,1)+' log units)':'')+'. '+(active?'Orange paths are discrete updates. Click to add an initial point.':'Arrows indicate score direction.');
    $('contour-canvas').classList.toggle('can-add',active);
    $('landscape-legend').hidden=state.stage===0;$('play-button').textContent=playing?'Pause':'Play';
    $('play-button').setAttribute('aria-pressed',String(playing));
    const converged=trajectories.filter(t=>t.status==='converged').length, escaped=trajectories.filter(t=>t.status==='escaped').length;
    $('playback-status').textContent=converged+escaped===trajectories.length&&trajectories.length?'Playback ends at update '+playbackEndIteration+': '+converged+' stationary tracks, '+escaped+' escaped.':'Playback shows up to '+playbackEndIteration+' actual updates; moving or oscillating tracks are retained.';
  }
  function canvasContext(canvas) {
    const rect=canvas.getBoundingClientRect(), dpr=Math.min(3,Math.max(2,window.devicePixelRatio||1)), w=Math.max(1,rect.width), h=Math.max(1,rect.height);
    if (w < 60 || h < 80) return null;
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=BG;ctx.fillRect(0,0,w,h);contexts.set(canvas.id,{ctx,w,h});return {ctx,w,h};
  }
  function line(ctx,points,color,width=1) { if(points.length<2)return;ctx.beginPath();ctx.moveTo(...points[0]);for(let i=1;i<points.length;i++)ctx.lineTo(...points[i]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke(); }
  function circle(ctx,x,y,r,color,stroke) { ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();} }
  function text(ctx,value,x,y,color=MUTED,size=12,align='left') { ctx.fillStyle=color;ctx.font=Math.max(12,size)+'px "Segoe UI",Arial,sans-serif';ctx.textAlign=align;ctx.fillText(value,x,y); }
  function arrow(ctx,a,b,color,width=1) { const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy);if(len<.1)return;line(ctx,[a,b],color,width);const ux=dx/len,uy=dy/len,r=4;ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-r*ux+r*.5*uy,b[1]-r*uy-r*.5*ux);ctx.lineTo(b[0]-r*ux-r*.5*uy,b[1]-r*uy+r*.5*ux);ctx.closePath();ctx.fillStyle=color;ctx.fill(); }
  function shellScales() { return [state.sigmaMin,Math.sqrt(state.sigmaMin*state.sigmaMax),state.sigmaMax]; }
  function drawShells() {
    const view=canvasContext($('shell-canvas'));if(!view)return;
    const {ctx,w,h}=view, scales=shellScales(), maxR=state.sigmaMax*(Math.sqrt(state.d)+2.7), scale=Math.min(w*.40,(h-65)*.44)/maxR,cx=w*.5,cy=(h-15)*.51;
    const radii=normals.map(row=>Math.sqrt(row.slice(0,state.d).reduce((a,v)=>a+v*v,0)));
    for(const sigma of M.geometricScales(state.sigmaMin,state.sigmaMax,N_SCALES)){ctx.beginPath();ctx.arc(cx,cy,sigma*Math.sqrt(state.d)*scale,0,Math.PI*2);ctx.strokeStyle='rgba(92,119,137,.16)';ctx.lineWidth=.7;ctx.stroke();}
    for(let k=2;k>=0;k--){const sigma=scales[k],rr=sigma*Math.sqrt(state.d)*scale;ctx.beginPath();ctx.arc(cx,cy,rr,0,Math.PI*2);ctx.strokeStyle=COLORS[k]+'75';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.stroke();ctx.setLineDash([]);for(let j=0;j<radii.length;j++){const r=radii[j]*sigma*scale;circle(ctx,cx+r*Math.cos(angles[j]),cy+r*Math.sin(angles[j]),1.6,COLORS[k]+'c0');}}
    circle(ctx,cx,cy,3.1,INK);text(ctx,'μ',cx+8,cy+5,INK,14);
    scales.forEach((s,k)=>{circle(ctx,12+k*w/3,h-27,3,COLORS[k]);text(ctx,'σ = '+format(s),23+k*w/3,h-23,INK,14);});
    const r=scales[1]*Math.sqrt(state.d)*scale,angle=-.45;arrow(ctx,[cx,cy],[cx+r*Math.cos(angle),cy+r*Math.sin(angle)],INK+'b0');text(ctx,'≈ σ√d',cx+r*.48,cy-r*.28-7,INK,12);
  }
  function drawProfiles() {
    const view=canvasContext($('profile-canvas'));if(!view)return;
    const {ctx,w,h}=view,scales=shellScales(),maxR=state.sigmaMax*(Math.sqrt(state.d)+3),left=38,right=w-17,plotW=right-left;
    const plots=[{top:30,bottom:h*.43,title:'Ambient point density / its peak',mass:false},{top:h*.57,bottom:h-30,title:'Radial probability density',mass:true}];
    for(const p of plots){const count=220,values=scales.map(s=>Array.from({length:count},(_,i)=>p.mass?M.radialMass(maxR*i/(count-1),s,state.d):M.ambientRelative(maxR*i/(count-1),s)));const maxY=p.mass?Math.max(...values.flat())*1.05:1.05;line(ctx,[[left,p.top+10],[left,p.bottom],[right,p.bottom]],'#cbd3d6',1);text(ctx,p.title,left,p.top-10,INK,12);text(ctx,p.mass?format(maxY,2):'1',left-8,p.top+15,MUTED,10,'right');text(ctx,'0',left-8,p.bottom+3,MUTED,10,'right');for(let k=0;k<3;k++){line(ctx,values[k].map((v,i)=>[left+plotW*i/(count-1),p.bottom-(p.bottom-p.top-10)*v/maxY]),COLORS[k],1.8);}for(let i=0;i<3;i++){let r=maxR*i/2;const x=left+plotW*i/2;text(ctx,format(r,1),x,p.bottom+16,MUTED,10,'center');}text(ctx,'radius r',right,p.bottom+30,MUTED,10,'right');}
  }
  function buildGrids() {
    const nextKey=JSON.stringify([state.d,state.sigmaMin,state.sigmaMax,config.centers]);
    if(surface&&contour&&gridConfigKey===nextKey)return;
    gridConfigKey=nextKey;
    gridRevision++;surfaceCache=null;contourCache=null;
    const n=55,points=[];let max=-Infinity,min=Infinity;
    for(let iy=0;iy<n;iy++)for(let ix=0;ix<n;ix++){const x=-LIMIT+2*LIMIT*ix/(n-1),y=-LIMIT+2*LIMIT*iy/(n-1),lp=M.evaluate(x,y,config).logDensity;points.push({x,y,lp});max=Math.max(max,lp);min=Math.min(min,lp);}
    config.centers.forEach(p=>{max=Math.max(max,M.evaluate(p[0],p[1],config).logDensity);});
    const size=128,image=new ImageData(size,size);
    const values=[];
    for(let iy=0;iy<size;iy++)for(let ix=0;ix<size;ix++){const x=-LIMIT+2*LIMIT*ix/(size-1),y=LIMIT-2*LIMIT*iy/(size-1);values.push(M.evaluate(x,y,config).logDensity);}
    for(const value of values){min=Math.min(min,value);max=Math.max(max,value);}
    const range=Math.max(1e-12,max-min);surface={n,points,min,max,range};
    const rgbA=[251,250,247],rgbB=[126,161,183];
    for(let iy=0;iy<size;iy++)for(let ix=0;ix<size;ix++){const index=iy*size+ix,q=Math.max(0,Math.min(1,(values[index]-min)/range));for(let c=0;c<3;c++)image.data[index*4+c]=rgbA[c]*(1-q*.7)+rgbB[c]*q*.7;image.data[index*4+3]=255;}
    const segments=[];
    for(let level=1;level<12;level++){const threshold=min+level*range/12;for(let iy=0;iy<size-1;iy++)for(let ix=0;ix<size-1;ix++){const indexes=[iy*size+ix,iy*size+ix+1,(iy+1)*size+ix+1,(iy+1)*size+ix],positions=[[ix,iy],[ix+1,iy],[ix+1,iy+1],[ix,iy+1]],crossings=[];for(let edge=0;edge<4;edge++){const next=(edge+1)%4,a=values[indexes[edge]],b=values[indexes[next]];if((a>=threshold)!==(b>=threshold)){const t=(threshold-a)/(b-a);crossings.push([(positions[edge][0]+t*(positions[next][0]-positions[edge][0]))/(size-1),(positions[edge][1]+t*(positions[next][1]-positions[edge][1]))/(size-1)]);}}if(crossings.length===2)segments.push([crossings[0],crossings[1]]);else if(crossings.length===4){segments.push([crossings[0],crossings[1]]);segments.push([crossings[2],crossings[3]]);}}}
    const raster=document.createElement('canvas');raster.width=size;raster.height=size;raster.getContext('2d').putImageData(image,0,0);contour={raster,segments};
  }
  function computeTrajectories() {
    const tolerance=POSITION_TOLERANCE*Math.max(1,state.sigmaMin),scoreTolerance=tolerance/(state.sigmaMin*state.sigmaMin);
    trajectories=baseStarts().map(start=>{const first=M.evaluate(...start,config),points=[{point:start,lp:first.logDensity,delta:0,out:false}],segments=[];let p=start,status='limit';for(let i=0;i<MAX_ITER;i++){const step=M.step(p,config,state.eta,state.rule),movement=Math.hypot(step.point[0]-p[0],step.point[1]-p[1]),residual=Math.hypot(step.after.barycenter[0]-step.point[0],step.after.barycenter[1]-step.point[1]),scoreResidual=Math.hypot(...step.after.score);const out=!step.isFinite||Math.hypot(...step.point)>20;points.push({point:step.point,lp:step.after.logDensity,delta:step.deltaLogDensity,out,movement,residual,scoreResidual});const clipped=clipToView(p,step.point);segments.push(clipped?clipped.map(q=>({point:q,lp:M.evaluate(...q,config).logDensity})):null);p=step.point;if(out){status='escaped';break;}if(residual<=tolerance&&scoreResidual<=scoreTolerance&&(movement<=tolerance||residual<=tolerance*.1)){status='converged';break;}}return {start,points,segments,status};});
    playbackEndIteration=Math.max(1,...trajectories.map(t=>t.points.length-1));
    playbackIterations=[0];let previous=0;
    for(let i=1;i<=playbackEndIteration;i++){
      const visibleChange=trajectories.some(t=>{const a=t.points[Math.min(previous,t.points.length-1)],b=t.points[Math.min(i,t.points.length-1)];return a.out!==b.out||(!a.out&&!b.out&&Math.hypot(b.point[0]-a.point[0],b.point[1]-a.point[1])>=.008);});
      if(visibleChange||i===playbackEndIteration){playbackIterations.push(i);previous=i;}
    }
    state.iteration=Math.min(state.iteration,playbackEndIteration);
  }
  function buildFarField() {
    const nextKey=[state.d,state.sigmaMin,state.sigmaMax].join(':');
    if(farField&&farFieldConfigKey===nextKey)return;
    farFieldConfigKey=nextKey;
    const baselineSigmaMax=Math.max(.8,state.sigmaMin),radius=1.8*Math.sqrt(state.d),maxRadius=2.6*Math.sqrt(state.d);
    const currentConfig=M.prepare({d:state.d,sigmaMin:state.sigmaMin,sigmaMax:state.sigmaMax,nScales:N_SCALES,centers:[[0,0]]});
    const referenceConfig=M.prepare({d:state.d,sigmaMin:state.sigmaMin,sigmaMax:baselineSigmaMax,nScales:N_SCALES,centers:[[0,0]]});
    const centerCurrent=M.evaluate(0,0,currentConfig).logDensity,centerReference=M.evaluate(0,0,referenceConfig).logDensity;
    const summarize=(prepared,center)=>{const e=M.evaluate(radius,0,prepared);return{logDensity:e.logDensity,relativeLogDensity:e.logDensity-center,scoreMagnitude:Math.hypot(...e.score),effectiveScale:1/Math.sqrt(e.beta)};};
    const current=summarize(currentConfig,centerCurrent),reference=summarize(referenceConfig,centerReference),samples=[];
    for(let i=0;i<181;i++){const r=maxRadius*i/180;samples.push({r,current:M.evaluate(r,0,currentConfig).logDensity-centerCurrent,reference:M.evaluate(r,0,referenceConfig).logDensity-centerReference});}
    farField={radius,maxRadius,baselineSigmaMax,currentSigmaMax:state.sigmaMax,nScales:N_SCALES,current,reference,samples,tailLift:current.relativeLogDensity-reference.relativeLogDensity};
  }
  function drawFarField() {
    const view=canvasContext($('far-field-canvas'));if(!view||!farField)return;const{ctx,w,h}=view;
    const left=70,right=w-17,top=24,bottom=h-42,minLog=Math.min(...farField.samples.map(p=>Math.min(p.current,p.reference))),range=Math.max(1,-minLog),xy=(r,lp)=>[left+r/farField.maxRadius*(right-left),top-lp/range*(bottom-top)];
    for(const fraction of [0,.5,1]){const lp=-range*fraction,p=xy(0,lp);line(ctx,[[left,p[1]],[right,p[1]]],'#d7e0e4',1);text(ctx,format(lp,0),left-9,p[1]+5,'#394a55',14,'right');}
    const referencePoints=farField.samples.map(p=>xy(p.r,p.reference));ctx.save();ctx.setLineDash([7,5]);line(ctx,referencePoints,ORANGE,2);ctx.restore();
    line(ctx,farField.samples.map(p=>xy(p.r,p.current)),BLUE,2.5);
    const probeX=xy(farField.radius,0)[0];ctx.save();ctx.setLineDash([3,4]);line(ctx,[[probeX,top],[probeX,bottom]],'#536e7f',1);ctx.restore();
    circle(ctx,...xy(farField.radius,farField.current.relativeLogDensity),4.4,BLUE,BG);circle(ctx,...xy(farField.radius,farField.reference.relativeLogDensity),4.4,ORANGE,BG);
    for(const r of [0,farField.radius,farField.maxRadius])text(ctx,format(r,1),xy(r,0)[0],bottom+22,'#394a55',14,'center');
    text(ctx,'log p(r) − log p(0)',left,15,INK,14);text(ctx,'radius r',right,h-3,'#394a55',14,'right');
    $('far-field-probe').textContent='Outer probe r = 1.8√d = '+format(farField.radius,2);
    $('far-field-scale').textContent=format(farField.current.effectiveScale,3)+' current';$('far-field-reference-scale').textContent=format(farField.reference.effectiveScale,3)+' reference';
    $('far-field-score').textContent=format(farField.current.scoreMagnitude,3)+' current';$('far-field-reference-score').textContent=format(farField.reference.scoreMagnitude,3)+' reference';
    $('far-field-current-label').textContent='Current σmax = '+format(state.sigmaMax,2);$('far-field-reference-label').textContent='Reference σmax = '+format(farField.baselineSigmaMax,2);
    const signed=(v)=>v>=0?'+'+format(v,3):'−'+format(-v,3);
    $('far-field-delta').textContent='At the same outer probe: effective scale changes by '+signed(farField.current.effectiveScale-farField.reference.effectiveScale)+' and score magnitude by '+signed(farField.current.scoreMagnitude-farField.reference.scoreMagnitude)+' versus the reference. The log-density tail is '+signed(farField.tailLift)+' units higher.';
  }
  function zHeight(lp) { return state.height==='density'?Math.exp(Math.min(0,lp-surface.max)):Math.max(0,Math.min(1,(lp-surface.min)/surface.range)); }
  function projection(w,h) {
    const scale=Math.min(w/13.8,h/9.9),top=24,bottom=h-35,heightScale=(h-87)*.66;
    const raw=(x,y,z)=>(x*Math.sin(yaw)+y*Math.cos(yaw))*scale*pitch-z*heightScale;
    const ys=[raw(-LIMIT,-LIMIT,0),raw(-LIMIT,LIMIT,0),raw(LIMIT,-LIMIT,0),raw(LIMIT,LIMIT,0)];
    if(surface)for(const p of surface.points)ys.push(raw(p.x,p.y,zHeight(p.lp)));
    const low=Math.min(...ys),high=Math.max(...ys),fit=Math.min(1,(bottom-top)/Math.max(1,high-low));
    return (x,y,z)=>[w*.5+(x*Math.cos(yaw)-y*Math.sin(yaw))*scale*fit,top+(raw(x,y,z)-low)*fit];
  }
  function drawSurface() {
    const view=canvasContext($('surface-canvas'));if(!view||!surface)return;const {ctx,w,h}=view;
    const project=projection(w,h),n=surface.n;
    const cacheKey=[gridRevision,w,h,$('surface-canvas').width,yaw,pitch,state.height].join(':');
    if(surfaceCache&&surfaceCache.key===cacheKey)ctx.drawImage(surfaceCache.image,0,0,w,h);
    else{
    for(let k=-4;k<=4;k+=2){line(ctx,[project(k,-LIMIT,0),project(k,LIMIT,0)],'#dfe5e7');line(ctx,[project(-LIMIT,k,0),project(LIMIT,k,0)],'#dfe5e7');}
    const cells=[];for(let y=0;y<n-1;y++)for(let x=0;x<n-1;x++){const p=surface.points[y*n+x],depth=p.x*Math.sin(yaw)+p.y*Math.cos(yaw);cells.push({x,y,depth});}cells.sort((a,b)=>a.depth-b.depth);
    for(const cell of cells){const p=[surface.points[cell.y*n+cell.x],surface.points[cell.y*n+cell.x+1],surface.points[(cell.y+1)*n+cell.x+1],surface.points[(cell.y+1)*n+cell.x]],z=p.reduce((s,v)=>s+zHeight(v.lp),0)/4,points=p.map(q=>project(q.x,q.y,zHeight(q.lp)));ctx.beginPath();ctx.moveTo(...points[0]);for(let i=1;i<4;i++)ctx.lineTo(...points[i]);ctx.closePath();const blend=.18+.72*Math.pow(z,.6),a=[234,240,241],b=[65,110,144];ctx.fillStyle='rgb('+a.map((v,i)=>Math.round(v*(1-blend)+b[i]*blend)).join(',')+')';ctx.fill();if(cell.x%5===0||cell.y%5===0){ctx.strokeStyle='rgba(50,90,117,.12)';ctx.lineWidth=.5;ctx.stroke();}}
    const ax=project(LIMIT,LIMIT,0),ay=project(-LIMIT,LIMIT,0);text(ctx,'x₁',ax[0]+8,ax[1]+13,MUTED,12);text(ctx,'x₂',ay[0]-5,ay[1]+17,MUTED,12);text(ctx,'d = '+state.d+' · 32 noise scales',w-9,h-10,MUTED,11,'right');
    const image=document.createElement('canvas');image.width=$('surface-canvas').width;image.height=$('surface-canvas').height;image.getContext('2d').drawImage($('surface-canvas'),0,0);surfaceCache={key:cacheKey,image};
    }
    if(state.stage===3){ctx.save();for(const t of trajectories){const last=Math.min(state.iteration,t.points.length-1);for(let i=1;i<=last;i++){const a=t.points[i-1],b=t.points[i];if(a.out)break;const segment=t.segments[i-1];if(segment)line(ctx,segment.map(p=>project(...p.point,zHeight(p.lp)+.015)),'rgba(199,120,73,.78)',1.2);if(b.out)break;}const q=t.points[last];if(!q.out&&Math.abs(q.point[0])<LIMIT&&Math.abs(q.point[1])<LIMIT){const pt=project(...q.point,zHeight(q.lp)+.02);circle(ctx,...pt,2.8,ORANGE,BG);}}ctx.restore();}
  }
  function contourTransform(w,h) { const size=Math.min(w-62,h-53),left=(w-size)/2,top=17;return {size,left,top,xy:p=>[left+(p[0]+LIMIT)/(2*LIMIT)*size,top+(LIMIT-p[1])/(2*LIMIT)*size]}; }
  function clipToView(a,b) {
    if (![...a,...b].every(Number.isFinite)) return null;
    let t0=0,t1=1;const dx=b[0]-a[0],dy=b[1]-a[1],ps=[-dx,dx,-dy,dy],qs=[a[0]+LIMIT,LIMIT-a[0],a[1]+LIMIT,LIMIT-a[1]];
    for(let k=0;k<4;k++){if(Math.abs(ps[k])<1e-15){if(qs[k]<0)return null;}else{const t=qs[k]/ps[k];if(ps[k]<0)t0=Math.max(t0,t);else t1=Math.min(t1,t);if(t0>t1)return null;}}
    return [[a[0]+t0*dx,a[1]+t0*dy],[a[0]+t1*dx,a[1]+t1*dy]];
  }
  function drawContour() {
    const canvas=$('contour-canvas'),view=canvasContext(canvas);if(!view||!contour)return;
    const {ctx,w,h}=view,{size,left,top,xy}=contourTransform(w,h),key=[gridRevision,w,h,canvas.width,state.stage].join(':');
    if(contourCache&&contourCache.key===key)ctx.drawImage(contourCache.image,0,0,w,h);
    else{
      ctx.drawImage(contour.raster,left,top,size,size);ctx.save();ctx.beginPath();ctx.rect(left,top,size,size);ctx.clip();ctx.beginPath();
      for(const segment of contour.segments){ctx.moveTo(left+segment[0][0]*size,top+segment[0][1]*size);ctx.lineTo(left+segment[1][0]*size,top+segment[1][1]*size);}
      ctx.strokeStyle='rgba(89,128,154,.48)';ctx.lineWidth=.65;ctx.stroke();
      if(state.stage!==3)for(let x=-3.5;x<=3.5;x+=.7)for(let y=-3.5;y<=3.5;y+=.7){const p=M.evaluate(x,y,config),length=Math.hypot(...p.score);if(length>.03){const pt=xy([x,y]),len=9+Math.min(6,Math.log1p(length));arrow(ctx,[pt[0]-p.score[0]/length*len*.4,pt[1]+p.score[1]/length*len*.4],[pt[0]+p.score[0]/length*len*.6,pt[1]-p.score[1]/length*len*.6],'rgba(65,110,144,.54)',.8);}}
      config.centers.forEach(p=>circle(ctx,...xy(p),3.2,BLUE,BG));ctx.restore();
      for(const x of [-4,0,4]){const p=xy([x,-LIMIT]);text(ctx,String(x),p[0],p[1]+17,MUTED,10,'center');}for(const y of [-4,0,4]){const p=xy([-LIMIT,y]);text(ctx,String(y),p[0]-9,p[1]+4,MUTED,10,'right');}
      text(ctx,'x₁',left+size,top+size+32,MUTED,11,'right');text(ctx,'x₂',left-20,top+5,MUTED,11);ctx.strokeStyle='#d7dfe3';ctx.lineWidth=1;ctx.strokeRect(left,top,size,size);
      const image=document.createElement('canvas');image.width=canvas.width;image.height=canvas.height;image.getContext('2d').drawImage(canvas,0,0);contourCache={key,image};
    }
    if(state.stage===3){ctx.save();ctx.beginPath();ctx.rect(left,top,size,size);ctx.clip();
      for(const t of trajectories){const last=Math.min(state.iteration,t.points.length-1);for(let i=1;i<=last;i++){const before=t.points[i-1],after=t.points[i];if(before.out)break;const segment=t.segments[i-1];if(segment)line(ctx,segment.map(p=>xy(p.point)),after.delta<-1e-9?'rgba(199,120,73,.92)':'rgba(199,120,73,.62)',1.2);if(after.out)break;}const start=xy(t.start);circle(ctx,...start,2.3,BG,ORANGE+'99');const q=t.points[last];if(!q.out)circle(ctx,...xy(q.point),3,ORANGE,BG);}
      config.centers.forEach(p=>circle(ctx,...xy(p),3.2,BLUE,BG));ctx.restore();
    }
  }
  function updateMetrics() {
    let downhill=0,outside=0,gains=[];for(const t of trajectories){const last=Math.min(state.iteration,t.points.length-1),q=t.points[last];for(let i=1;i<=last;i++)if(t.points[i].delta<-1e-8)downhill++;if(q.out||Math.abs(q.point[0])>LIMIT||Math.abs(q.point[1])>LIMIT)outside++;if(Number.isFinite(q.lp))gains.push(q.lp-t.points[0].lp);}
    gains.sort((a,b)=>a-b);const median=gains.length?gains[Math.floor(gains.length/2)]:NaN;
    $('start-count').textContent=trajectories.length;$('downhill-count').textContent=downhill;$('outside-count').textContent=outside;$('density-gain').textContent=format(median);
    $('downhill-count').classList.toggle('warning',downhill>0);$('outside-count').classList.toggle('warning',outside>0);
    if(state.stage===0){const exponent=state.d-2;$('reading-line').innerHTML=exponent===0?'At <strong>d = 2</strong>, doubling σ multiplies shell volume and weighted mass by four: their scaling balances. Each Gaussian still has its highest point density at the center.':'Double the scale: <strong>2<sup>'+state.d+'</sup> times the shell volume</strong>, but only four times the weighted mass. The effective mean shell-density ratio is 2<sup>−'+exponent+'</sup>.';}
    else if(state.stage===1){const stab=M.singleCenterStability(config);$('reading-line').innerHTML='One center, one summit. Near the center, ordinary Euler ascent is locally stable for <strong>η &lt; '+format(stab.localRawEulerLimit,3)+'</strong> with these scales. This limit changes with the terrain.';}
    else if(state.stage===2){$('reading-line').innerHTML='The score combines all center–scale contributions. Overlap allows <strong>several nearby centers</strong> to influence a single direction.';}
    else if(state.rule==='normalized'&&state.eta>=2){$('reading-line').innerHTML='<strong>η ≥ 2 is outside the guaranteed ascent range</strong> of the normalized update. Density may decrease or trajectories may oscillate or escape.';}
    else if(downhill){$('reading-line').innerHTML='The step can outrun the slope. <strong>'+downhill+' density-decreasing updates</strong> have occurred by this iteration. Reduce η or try the normalized update.';}
    else{$('reading-line').innerHTML=state.rule==='normalized'?'The normalized update follows a weighted average of centers. <strong>0 &lt; η &lt; 2</strong> is an ascent range for this exact mixture.':'The landscape is fixed. Randomness enters through the initial locations; <strong>the displayed updates are deterministic</strong>.';}
    const single=M.prepare({d:state.d,sigmaMin:state.sigmaMin,sigmaMax:state.sigmaMax,nScales:N_SCALES,centers:[[0,0]]}),stab=M.singleCenterStability(single);$('stability-note').textContent='For the single-center mixture at these settings, the exact local raw-Euler limit at the summit is η < '+format(stab.localRawEulerLimit,4)+'. It is a local result, not a global step-size guarantee for the multi-center landscape.';
  }
  function renderOnly(playbackOnly=false) { syncControls();if(state.stage===0){drawShells();drawProfiles();}else{drawSurface();drawContour();if(!playbackOnly)drawFarField();}updateMetrics();setTimeout(reportHeight,0); }
  function rebuild() { config=chooseConfig();buildFarField();if(state.stage!==0){buildGrids();computeTrajectories();}renderOnly(); }
  function scheduleRebuild() { if(pending)return;pending=true;$('play-button').disabled=true;$('playback-status').textContent='Updating the terrain and trajectories…';setTimeout(()=>{pending=false;rebuild();},0); }
  function stop() { playing=false;if(playbackTimer!==null){clearTimeout(playbackTimer);playbackTimer=null;}$('play-button').textContent='Play';$('play-button').setAttribute('aria-pressed','false'); }
  function setParameters(patch) {
    if (patch.mode) patch={...patch,rule:patch.mode};
    stop();state={...state,...patch};state.d=Math.max(2,Math.min(128,Math.round(state.d)));state.sigmaMin=Math.max(.12,Math.min(1,state.sigmaMin));state.sigmaMax=Math.max(.8,Math.min(2.5,state.sigmaMax));if(state.sigmaMin>state.sigmaMax)state.sigmaMax=state.sigmaMin;const [a,b]=etaRange();state.eta=Math.max(a,Math.min(b,state.eta));state.iteration=0;syncControls();scheduleRebuild();
  }
  function setStage(index) { stop();state.stage=Math.max(0,Math.min(3,Math.round(index)));state.iteration=0;syncControls();scheduleRebuild(); }
  function setIteration(n) { state.iteration=Math.max(0,Math.min(playbackEndIteration,Math.round(n)));renderOnly(true); }
  document.querySelectorAll('[data-stage]').forEach(el=>el.addEventListener('click',()=>setStage(+el.dataset.stage)));
  document.querySelectorAll('[data-height]').forEach(el=>el.addEventListener('click',()=>{state.height=el.dataset.height;renderOnly();}));
  $('dimension').addEventListener('input',e=>setParameters({d:+e.target.value}));
  $('sigma-min').addEventListener('input',e=>setParameters({sigmaMin:+e.target.value}));
  $('sigma-max').addEventListener('input',e=>{const sigmaMax=+e.target.value;setParameters({sigmaMax,sigmaMin:Math.min(state.sigmaMin,sigmaMax)});});
  $('step-size').addEventListener('input',e=>setParameters({eta:sliderToEta(+e.target.value)}));
  $('gradient-rule').addEventListener('change',e=>setParameters({rule:e.target.value,eta:e.target.value==='raw'?.03:.2}));
  $('ascent-centers').addEventListener('change',e=>setParameters({target:e.target.value}));
  $('iteration').addEventListener('input',e=>{stop();setIteration(+e.target.value);});
  $('step-button').addEventListener('click',()=>{stop();setIteration(state.iteration>=playbackEndIteration?0:state.iteration+1);});
  $('restart-button').addEventListener('click',()=>{stop();setIteration(0);});
  $('play-button').addEventListener('click',()=>{if(playing){stop();return;}if(pending)return;if(state.iteration>=playbackEndIteration)state.iteration=0;playing=true;playbackFrom=state.iteration;playbackStarted=performance.now();playbackFrames=[playbackFrom,...playbackIterations.filter(n=>n>playbackFrom)];if(playbackFrames[1]!==playbackFrom+1)playbackFrames.splice(1,0,playbackFrom+1);playbackDuration=Math.min(6000,Math.max(1800,(playbackFrames.length-1)*85));setIteration(Math.min(playbackEndIteration,state.iteration+1));if(state.iteration>=playbackEndIteration){stop();return;}playbackTimer=setTimeout(animate,33);});
  $('reset-button').addEventListener('click',()=>{extraStarts=[];yaw=.72;state={...defaults};setParameters({});});
  document.querySelectorAll('[data-preset]').forEach(el=>el.addEventListener('click',()=>{extraStarts=[];if(el.dataset.preset==='thin'){setParameters({stage:0,d:128,sigmaMin:.32,sigmaMax:1.35,rule:'raw',eta:.03});}else if(el.dataset.preset==='overlap'){setParameters({stage:2,d:2,sigmaMin:.55,sigmaMax:1.5,rule:'raw',eta:.03});}else if(el.dataset.preset==='broad'){setParameters({stage:1,d:16,sigmaMin:.32,sigmaMax:2.5,rule:'raw',eta:.03,height:'log'});}else{setParameters({stage:3,d:16,sigmaMin:.18,sigmaMax:1.35,rule:'raw',eta:.12,target:'single'});state.iteration=12;}}));
  let drag=null;
  $('surface-canvas').addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch};e.target.setPointerCapture(e.pointerId);});
  $('surface-canvas').addEventListener('pointermove',e=>{if(!drag)return;yaw=drag.yaw+(e.clientX-drag.x)*.008;pitch=Math.max(.22,Math.min(.75,drag.pitch+(e.clientY-drag.y)*.002));drawSurface();});
  $('surface-canvas').addEventListener('pointerup',()=>drag=null);
  $('surface-canvas').addEventListener('pointercancel',()=>drag=null);
  $('surface-canvas').addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();yaw+=e.key==='ArrowLeft'?-.12:.12;drawSurface();}});
  $('contour-canvas').addEventListener('click',e=>{if(state.stage!==3||extraStarts.length>=12)return;const rect=e.currentTarget.getBoundingClientRect(),{size,left,top}=contourTransform(rect.width,rect.height),x=e.clientX-rect.left,y=e.clientY-rect.top;if(x<left||x>left+size||y<top||y>top+size)return;extraStarts.push([(x-left)/size*2*LIMIT-LIMIT,LIMIT-(y-top)/size*2*LIMIT]);stop();state.iteration=0;computeTrajectories();renderOnly();});
  const resize=new ResizeObserver(()=>{if(resizeTimer!==null)clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{resizeTimer=null;if(config)renderOnly();},0);});canvases.forEach(c=>resize.observe(c));
  function animate() { playbackTimer=null;if(!playing||state.stage!==3)return;const elapsed=Math.max(0,performance.now()-playbackStarted),fraction=Math.min(1,elapsed/Math.max(1,playbackDuration)),next=playbackFrames[Math.min(playbackFrames.length-1,1+Math.floor((playbackFrames.length-1)*fraction))];if(next!==state.iteration)setIteration(next);if(fraction>=1||state.iteration>=playbackEndIteration){stop();return;}playbackTimer=setTimeout(animate,33); }
  new ResizeObserver(reportHeight).observe(document.body);
  window.landscapeDemo={getState:()=>({...state,nScales:N_SCALES,mode:state.rule,pending,playing,playbackEndIteration,meaningfulFrameCount:playbackIterations.length,playbackDurationMs:playbackDuration,playbackIterations:playbackIterations.slice(),stationarityTolerance:POSITION_TOLERANCE*Math.max(1,state.sigmaMin),starts:baseStarts(),trajectoryCount:trajectories.length,gridMaximum:surface?surface.max:null,gridMinimum:surface?surface.min:null,displayedLogRange:surface?surface.range:null,farField:farField?{radius:farField.radius,maxRadius:farField.maxRadius,baselineSigmaMax:farField.baselineSigmaMax,currentSigmaMax:farField.currentSigmaMax,nScales:farField.nScales,current:{...farField.current},reference:{...farField.reference},tailLift:farField.tailLift}:null,trajectories:trajectories.map(t=>({start:t.start,status:t.status,points:t.points.map(p=>({point:p.point.slice(),logDensity:p.lp,delta:p.delta,out:p.out,residual:p.residual,scoreResidual:p.scoreResidual}))}))}),setParameters,setStage,setIteration,reset:()=>{$('reset-button').click();}};
  rebuild();
})();
