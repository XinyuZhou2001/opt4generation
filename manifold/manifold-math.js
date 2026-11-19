/* Exact finite-mixture mechanics for the local-manifold explorable.
 * Requires GaussianLandscapeMath (../interactive/landscape-math.js).
 * All 82 centers lie on (u, kappa*u*u/2) in an invariant 2D slice of R^3.
 * The reference curve is the whole parabola, not a fitted curve through an
 * output. kappa is its curvature AT THE VERTEX; curvature decreases away.
 * A fixed log-spaced noise grid has sigmaMax=3 by default. The empirical
 * mixture uses sigma^2 weights. Finite numerical outputs are not a theorem
 * of population generalization; tolerances below are explicit diagnostics.
 *
 * prepare({curvature:.1,spacing:.8,sigmaMin:.2,sigmaMax:3,nScales:32,d:3})
 * evaluate(x,y,cfg) -> exact score, beta, barycenter, q/driftWeights,
 *   densityResponsibilities, topShare, effectiveContributors, logDensity.
 * pointAt(s,cfg) -> [x,y] at signed arclength s from the vertex.
 * project([x,y],cfg) -> {point,u,s,distance,signedDistance,normal,tie,curvature}.
 * diagnose(point,cfg,tolerance?) -> distance and contributor diagnostics.
 * simulate(start,cfg,{eta:.35,maxSteps:1200,tolerance:1e-6*spacing}) ->
 *   {points:[{point,logDensity,residual,trainingDistance,normalError,
 *             topShare,effectiveContributors}],endpoint,converged,residual,
 *    updates,minimumLogDensityIncrement,diagnostics,tolerance,eta}.
 */
(function attach(root,factory){
  'use strict';
  const G=typeof module==='object'&&module.exports?require('../interactive/landscape-math.js'):root.GaussianLandscapeMath;
  const api=factory(G);
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.ManifoldMath=api;
})(typeof globalThis!=='undefined'?globalThis:this,function makeAPI(G){
  'use strict';
  if(!G)throw new Error('GaussianLandscapeMath must be loaded first');

  function arcLength(u,kappa){
    if(Math.abs(kappa)<1e-10)return u;
    const v=kappa*u;
    return .5*(u*Math.sqrt(1+v*v)+Math.asinh(v)/kappa);
  }
  function parameterAt(s,kappa){
    if(Math.abs(kappa)<1e-10)return s;
    let u=Math.sign(s)*Math.min(Math.abs(s),Math.sqrt(2*Math.abs(s)/kappa));
    for(let n=0;n<24;n++){
      const residual=arcLength(u,kappa)-s;
      const delta=residual/Math.sqrt(1+kappa*kappa*u*u);
      u-=delta;if(Math.abs(delta)<1e-13*(1+Math.abs(u)))break;
    }
    return u;
  }
  function pointAt(s,cfg){const u=parameterAt(s,cfg.curvature);return [u,.5*cfg.curvature*u*u];}
  function prepare(input){
    input=input||{};
    const curvature=input.curvature===undefined?.1:Number(input.curvature),spacing=input.spacing===undefined?.8:Number(input.spacing);
    if(!Number.isFinite(curvature)||curvature<0||curvature>10)throw new RangeError('curvature must be between 0 and 10');
    if(!Number.isFinite(spacing)||spacing<=0)throw new RangeError('spacing must be positive');
    const halfCount=input.halfCount===undefined?20:input.halfCount;
    if(!Number.isInteger(halfCount)||halfCount<2||halfCount>200)throw new RangeError('halfCount must be an integer from 2 to 200');
    const centers=[],centerMeta=[];
    for(let cell=-halfCount;cell<=halfCount;cell++)for(const side of [-1,1]){
      const s=spacing*(cell+side/6),u=parameterAt(s,curvature);
      centers.push([u,.5*curvature*u*u]);centerMeta.push(Object.freeze({s,u,cell,side}));
    }
    const base=G.prepare({d:input.d===undefined?3:input.d,sigmaMin:input.sigmaMin===undefined?.2:input.sigmaMin,
      sigmaMax:input.sigmaMax===undefined?3:input.sigmaMax,nScales:input.nScales===undefined?32:input.nScales,centers});
    return Object.freeze({...base,curvature,spacing,halfCount,reach:curvature>0?1/curvature:Infinity,
      centerMeta:Object.freeze(centerMeta),_manifoldPrepared:true});
  }
  function invalid(n){return {logDensity:-Infinity,score:[NaN,NaN],beta:NaN,barycenter:[NaN,NaN],q:Array(n).fill(0),driftWeights:Array(n).fill(0),densityResponsibilities:Array(n).fill(0),topShare:NaN,topIndex:-1,effectiveContributors:NaN,isFinite:false};}
  function evaluate(x,y,cfg){
    if(!cfg||!cfg._manifoldPrepared)throw new TypeError('Expected a prepared manifold configuration');
    const centers=cfg.centers,n=centers.length;
    if(!Number.isFinite(x)||!Number.isFinite(y))return invalid(n);
    const distances=new Float64Array(n);let maximum=-Infinity;
    for(let j=0;j<n;j++){
      const dx=x-centers[j][0],dy=y-centers[j][1],r2=dx*dx+dy*dy;distances[j]=r2;
      for(let i=0;i<cfg.nScales;i++)maximum=Math.max(maximum,cfg.logCoefficients[i]-.5*r2*cfg.invVariances[i]);
    }
    if(!Number.isFinite(maximum))return invalid(n);
    const q=new Array(n),densityResponsibilities=new Array(n);let mass=0,precision=0,bx=0,by=0;
    for(let j=0;j<n;j++){
      let mj=0,pj=0;
      for(let i=0;i<cfg.nScales;i++){
        const a=Math.exp(cfg.logCoefficients[i]-.5*distances[j]*cfg.invVariances[i]-maximum);
        mj+=a;pj+=a*cfg.invVariances[i];
      }
      q[j]=pj;densityResponsibilities[j]=mj;mass+=mj;precision+=pj;bx+=pj*centers[j][0];by+=pj*centers[j][1];
    }
    bx/=precision;by/=precision;const beta=precision/mass;let concentration=0,topShare=-Infinity,topIndex=-1;
    for(let j=0;j<n;j++){
      q[j]/=precision;densityResponsibilities[j]/=mass;concentration+=q[j]*q[j];
      if(q[j]>topShare){topShare=q[j];topIndex=j;}
    }
    const score=[beta*(bx-x),beta*(by-y)],logDensity=maximum+Math.log(mass)-cfg.logNormalizer;
    const isFinite=Number.isFinite(logDensity)&&Number.isFinite(beta)&&score.every(Number.isFinite);
    return {logDensity,score,beta,barycenter:[bx,by],q,driftWeights:q,densityResponsibilities,
      centerResponsibilities:densityResponsibilities,topShare,topIndex,effectiveContributors:1/concentration,isFinite};
  }
  function logDensity(x,y,cfg){
    let maximum=-Infinity,sum=0;const terms=[];
    for(const center of cfg.centers){const dx=x-center[0],dy=y-center[1],r2=dx*dx+dy*dy;
      for(let i=0;i<cfg.nScales;i++){const t=cfg.logCoefficients[i]-.5*r2*cfg.invVariances[i];terms.push(t);maximum=Math.max(maximum,t);}}
    for(const t of terms)sum+=Math.exp(t-maximum);
    return maximum+Math.log(sum)-cfg.logNormalizer;
  }
  function cubicRoots(p,q){
    const discriminant=q*q/4+p*p*p/27,scale=1+Math.abs(q*q/4)+Math.abs(p*p*p/27),roots=[];
    if(p<0&&discriminant<=1e-14*scale){
      const r=2*Math.sqrt(-p/3),arg=Math.max(-1,Math.min(1,(3*q/(2*p))*Math.sqrt(-3/p))),theta=Math.acos(arg)/3;
      for(let j=0;j<3;j++)roots.push(r*Math.cos(theta+2*Math.PI*j/3));
    }else{
      const sd=Math.sqrt(Math.max(0,discriminant));roots.push(Math.cbrt(-q/2+sd)+Math.cbrt(-q/2-sd));
    }
    return roots;
  }
  function project(point,cfg){
    const x=point[0],y=point[1],k=cfg.curvature;
    if(!Number.isFinite(x)||!Number.isFinite(y))return {point:[NaN,NaN],u:NaN,s:NaN,distance:Infinity,signedDistance:NaN,normal:[NaN,NaN],tie:false,curvature:NaN};
    if(k<1e-10)return {point:[x,0],u:x,s:x,distance:Math.abs(y),signedDistance:y,normal:[0,1],tie:false,curvature:0};
    // Critical points of squared Euclidean distance solve
    // (k^2/2)u^3 + (1-k*y)u - x = 0. Compare ALL real roots.
    const p=2*(1-k*y)/(k*k),q=-2*x/(k*k),roots=cubicRoots(p,q);
    const candidates=[];
    for(let u of roots){
      for(let n=0;n<5;n++){const f=.5*k*k*u*u*u+(1-k*y)*u-x,df=1-k*y+1.5*k*k*u*u;if(Math.abs(df)<1e-13)break;const du=f/df;u-=du;if(Math.abs(du)<1e-13*(1+Math.abs(u)))break;}
      const yy=.5*k*u*u,d2=(x-u)*(x-u)+(y-yy)*(y-yy);candidates.push({u,yy,d2});
    }
    candidates.sort((a,b)=>a.d2-b.d2||a.u-b.u);const best=candidates[0];
    const tie=candidates.slice(1).some(c=>Math.abs(c.u-best.u)>1e-7&&Math.abs(c.d2-best.d2)<1e-10*(1+best.d2));
    const norm=Math.sqrt(1+k*k*best.u*best.u),normal=[-k*best.u/norm,1/norm];
    return {point:[best.u,best.yy],u:best.u,s:arcLength(best.u,k),distance:Math.sqrt(Math.max(0,best.d2)),
      signedDistance:(x-best.u)*normal[0]+(y-best.yy)*normal[1],normal,tie,curvature:k/Math.pow(1+k*k*best.u*best.u,1.5)};
  }
  function nearestTraining(point,cfg){
    let distance=Infinity,index=-1;
    for(let j=0;j<cfg.centers.length;j++){const c=cfg.centers[j],r=Math.hypot(point[0]-c[0],point[1]-c[1]);if(r<distance){distance=r;index=j;}}
    return {distance,index,point:cfg.centers[index]};
  }
  function diagnose(point,cfg,tolerance){
    const ev=evaluate(...point,cfg),projection=project(point,cfg),training=nearestTraining(point,cfg);
    const residual=Math.hypot(ev.barycenter[0]-point[0],ev.barycenter[1]-point[1]);
    const margin=cfg.spacing/12,manifoldMargin=cfg.spacing/10,tol=tolerance===undefined?1e-6*cfg.spacing:tolerance;
    return {...ev,residual,trainingDistance:training.distance,trainingIndex:training.index,normalError:projection.distance,
      projection,cell:Math.round(projection.s/cfg.spacing),trainingRatio:training.distance/cfg.spacing,
      normalRatio:projection.distance/cfg.spacing,margin,trainingMargin:margin,manifoldMargin,nearTraining:training.distance<margin,
      nearManifold:projection.distance<=manifoldMargin,stationary:residual<=tol,
      diagnostic:'Near training uses distance < ell/12; near the curve uses distance <= ell/10. These are illustrative diagnostics, not memorization or generalization guarantees.'};
  }
  function simulate(start,cfg,options){
    options=options||{};const eta=options.eta===undefined?.35:options.eta,maxSteps=options.maxSteps===undefined?1200:options.maxSteps;
    const tolerance=options.tolerance===undefined?1e-6*cfg.spacing:options.tolerance;
    if(!(eta>0&&eta<2))throw new RangeError('This explorable uses normalized Euler with 0 < eta < 2');
    if(!Number.isInteger(maxSteps)||maxSteps<0||maxSteps>20000)throw new RangeError('maxSteps must be an integer from 0 to 20000');
    let point=start.slice(),ev=evaluate(...point,cfg),residual=Math.hypot(ev.barycenter[0]-point[0],ev.barycenter[1]-point[1]);
    let minimumLogDensityIncrement=Infinity,converged=residual<=tolerance,updates=0;const points=[];
    function record(delta){const proj=project(point,cfg),tr=nearestTraining(point,cfg);points.push({point:point.slice(),logDensity:ev.logDensity,residual,
      delta,trainingDistance:tr.distance,normalError:proj.distance,topShare:ev.topShare,effectiveContributors:ev.effectiveContributors});}
    record(0);
    while(updates<maxSteps&&!converged){
      const next=[point[0]+eta*(ev.barycenter[0]-point[0]),point[1]+eta*(ev.barycenter[1]-point[1])];
      const after=evaluate(...next,cfg),delta=after.logDensity-ev.logDensity;minimumLogDensityIncrement=Math.min(minimumLogDensityIncrement,delta);
      point=next;ev=after;residual=Math.hypot(ev.barycenter[0]-point[0],ev.barycenter[1]-point[1]);updates++;record(delta);
      if(!ev.isFinite)break;converged=residual<=tolerance;
    }
    if(!Number.isFinite(minimumLogDensityIncrement))minimumLogDensityIncrement=0;
    const diagnostics=diagnose(point,cfg,tolerance);
    return {start:start.slice(),points,endpoint:point.slice(),converged,residual,updates,minimumLogDensityIncrement,diagnostics,tolerance,eta};
  }
  const presets=Object.freeze({capture:Object.freeze({curvature:.1,spacing:.8,sigmaMin:.04}),
    shared:Object.freeze({curvature:.1,spacing:.8,sigmaMin:.2}),
    bias:Object.freeze({curvature:.6,spacing:.8,sigmaMin:1}),
    positiveFloor:Object.freeze({curvature:.02,spacing:1.3,sigmaMin:.5}),
    curvedFloor:Object.freeze({curvature:.8,spacing:1.3,sigmaMin:.5})});
  return Object.freeze({prepare,evaluate,logDensity,arcLength,parameterAt,pointAt,project,nearestTraining,diagnose,simulate,presets});
});
