/* Guided generation on one connected manifold.
 * The circle is split into a constraint-satisfying arc and remaining data.
 * The fixed prior is an exact sigma^2-weighted Gaussian mixture around a
 * dense circular reference in an invariant two-dimensional slice of R^8.
 * Off-band score errors are conceptual, not measured neural-network errors.
 */
(function(root,factory){
  const G=typeof module==='object'&&module.exports?require('../interactive/landscape-math.js'):root.GaussianLandscapeMath;
  const api=factory(G);if(typeof module==='object'&&module.exports)module.exports=api;root.GuidanceMath=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(G){
  'use strict';
  if(!G)throw new Error('GaussianLandscapeMath must be loaded first');
  const d=8,R=2,sigmaMin=.06,sigmaMax=2,nScales=32,baseSteps=80,totalTime=32;
  const bandFactor=2.5,manifoldTolerance=2.5*sigmaMin,minX=.8,interiorX=1.3,guidanceGain=10,transverseDamping=.2;
  const oldPoint=Object.freeze([-3,3.5]);
  const centers=Object.freeze(Array.from({length:96},(_,i)=>Object.freeze([R*Math.cos(i*2*Math.PI/96),R*Math.sin(i*2*Math.PI/96)])));
  const mixture=G.prepare({d,centers,sigmaMin,sigmaMax,nScales});
  const softplus=z=>z>30?z:z< -30?Math.exp(z):Math.log1p(Math.exp(z));
  const sigmoid=z=>z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z));
  const norm=p=>Math.hypot(p[0],p[1]);
  function validate(parameters){
    const lambda=Number(parameters?.lambda??9),alpha=Number(parameters?.alpha??1);
    if(!Number.isFinite(lambda)||lambda<0||lambda>10||!Number.isFinite(alpha)||alpha<.25||alpha>4)throw new RangeError('lambda must be in [0,10] and alpha in [.25,4]');
    return{lambda,alpha};
  }
  function geometry(point){
    const radius=norm(point),manifoldDistance=Math.abs(radius-R),onManifold=manifoldDistance<=manifoldTolerance+1e-10;
    const constraintFeasible=point[0]>=minX;
    return{radius,manifoldDistance,onManifold,constraintFeasible,inConstraintSide:constraintFeasible,
      constraintSatisfied:onManifold&&constraintFeasible,inConstrainedManifold:onManifold&&constraintFeasible,
      projection:radius>1e-12?point.map(v=>R*v/radius):[R,0]};
  }
  // A smooth one-sided penalty favors the interior of the highlighted arc.
  // It specifies a region preference, not a point on the manifold.
  function penalty(point){return .5*guidanceGain*softplus(interiorX-point[0])**2;}
  function taskAt(point,lambda){const z=interiorX-point[0];return[lambda*guidanceGain*softplus(z)*sigmoid(z),0];}
  function objective(point,lambda){return G.evaluate(...point,mixture).logDensity-lambda*penalty(point);}
  function scaleWeights(point){
    const sigmas=mixture.sigmas,logs=[];
    for(const center of centers){const distance2=(point[0]-center[0])**2+(point[1]-center[1])**2;
      for(const s of sigmas)logs.push((2-d)*Math.log(s)-distance2/(2*s*s));}
    const mx=Math.max(...logs),raw=logs.map(v=>Math.exp(v-mx)),sum=raw.reduce((a,b)=>a+b,0),w=raw.map(v=>v/sum);
    const beta=w.reduce((a,v,k)=>a+v/(sigmas[k%sigmas.length]**2),0),q=w.map((v,k)=>v/(sigmas[k%sigmas.length]**2*beta));
    const driftWeights=sigmas.map((_,i)=>{let s=0;for(let j=0;j<centers.length;j++)s+=q[j*sigmas.length+i];return s;});
    const densityWeights=sigmas.map((_,i)=>{let s=0;for(let j=0;j<centers.length;j++)s+=w[j*sigmas.length+i];return s;});
    const dominantIndex=driftWeights.indexOf(Math.max(...driftWeights));
    return{sigmas:sigmas.slice(),densityWeights,driftWeights,beta,effectiveSigma:1/Math.sqrt(beta),dominantSigma:sigmas[dominantIndex]};
  }
  function taskCurvature(point){const z=interiorX-point[0],s=sigmoid(z);return guidanceGain*(s*s+softplus(z)*s*(1-s));}
  // Full lambda-curvature scaling would almost cancel the lambda in the
  // task force. This positive, step-aware damping preserves visible force
  // differences on fine steps and strengthens stabilization on coarse ones.
  function guidanceDamping(lambda,h){return Math.sqrt(lambda*(1+lambda*h*h)/(1+h*h));}
  // On the fixed landscape, a small task force should steer the approach
  // before the steep near-manifold prior makes tangential motion slow.
  // This positive numerical damping balances that approach continuously;
  // it changes neither score nor task gradient and vanishes from the
  // unguided update because taskScale is zero. Strong-guidance damping
  // smoothly returns to the existing transverse coefficient.
  function stateOnlyTransverseDamping(lambda){return transverseDamping+3.5*Math.exp(-((lambda/2)**2));}
  function sample(point,k,kind,lambda,errorStrength,schedule,configs,h){
    const scheduledSigma=schedule[k],config=kind==='conditional'?configs[k]:mixture;
    const field=G.evaluate(...point,config),geo=geometry(point),weights=scaleWeights(point);
    const availableSigma=kind==='conditional'?scheduledSigma:sigmaMax;
    const gap=Math.max(0,geo.manifoldDistance/(bandFactor*availableSigma)-1);
    // A C2 transition lets extrapolation error grow across the edge of the
    // training band instead of switching the direction at one threshold.
    const transition=Math.min(1,gap/.16);
    const activation=errorStrength*transition**3*(10+transition*(-15+6*transition));
    const unit=point.map(v=>v/Math.max(geo.radius,1e-12));
    // Conceptual off-band extrapolation, not measured network error or
    // sampling noise. Smooth spatial/label-dependent errors remain
    // correlated while changing direction as the assigned label shrinks;
    // they do not settle onto a second, apparently reliable circular flow.
    // The radial envelope only keeps the drawing in view. This same field
    // is available outside either description's training coverage, and the
    // guidance term is still added separately at every update below.
    const phase=Math.log(sigmaMax/availableSigma),x=point[0],y=point[1];
    const angle=1.2*Math.exp(-(((phase-.18)/.22)**2))-2+4.3*phase+.45*Math.sin(3*phase)+.2*Math.sin(x)+4*x+.4*y;
    const radial=.8*(4.4-geo.radius)+.35*Math.sin(4.3*phase+.2*y);
    const forcing=[1.5*Math.cos(angle),1.5*Math.sin(angle)];
    const wandering=unit.map((v,i)=>radial*v+forcing[i]);
    const error=field.score.map((v,i)=>activation*(field.beta*wandering[i]-v));
    const task=taskAt(point,lambda),direction=field.score.map((v,i)=>v+task[i]+error[i]);
    // The conditioned prior follows the VE probability-flow clock rather
    // than optimizing each broad-noise density to its (interior) mode.
    // At lambda=0 its Euler displacement is exactly
    // .5 * (sigma_k^2 - sigma_{k+1}^2) * score(x_k, sigma_k).
    // The terminal derivative is only a finite, unused query record.
    const noiseRatio=schedule[1]/schedule[0];
    const nextSigma=k+1<schedule.length?schedule[k+1]:scheduledSigma*noiseRatio;
    const noiseVarianceStep=.5*(scheduledSigma**2-nextSigma**2);
    const mobility=kind==='conditional'?noiseVarianceStep/h:1/field.beta;
    const inverseMobility=1/mobility;
    // Both solvers retain the same score-plus-guidance form, with a positive
    // diagonal damping that leaves vertical manifold return active.
    const guidanceDampingScale=guidanceDamping(lambda,h),taskScale=guidanceDampingScale*taskCurvature(point);
    const transverseDampingScale=kind==='unconditioned'?stateOnlyTransverseDamping(lambda):transverseDamping;
    const normalizers=[inverseMobility+taskScale,inverseMobility+transverseDampingScale*taskScale];
    const normalizer=normalizers[0],velocity=direction.map((v,i)=>v/normalizers[i]);
    const selectedBand={inner:Math.max(0,R-bandFactor*weights.dominantSigma),outer:R+bandFactor*weights.dominantSigma,halfWidth:bandFactor*weights.dominantSigma,schematic:true};
    return{point:point.slice(),iteration:k,time:k*h,...geo,scheduledSigma,assignedSigma:scheduledSigma,
      prior:field.score.slice(),score:field.score.slice(),task,error,direction,velocity,normalizer,normalizers,mobility,inverseMobility,noiseVarianceStep,guidanceDampingScale,transverseDampingScale,beta:field.beta,
      logDensity:field.logDensity,objective:objective(point,lambda),weights,dominantSigma:weights.dominantSigma,
      effectiveSigma:weights.effectiveSigma,implicitSigma:weights.effectiveSigma,geometricSigma:geo.manifoldDistance/Math.sqrt(d-1),
      trainingCovered:gap===0,coverageGap:gap,mismatch:Math.log1p(gap),errorActivation:activation,
      trainingBand:{inner:Math.max(0,R-bandFactor*availableSigma),outer:R+bandFactor*availableSigma,halfWidth:bandFactor*availableSigma,schematic:true},
      dominantBand:selectedBand,dominantShell:selectedBand,nearestDistance:geo.manifoldDistance,taskApplied:true};
  }
  function guidedComparison(parameters,options){
    const{lambda,alpha}=validate(parameters);options=options||{};
    const nominalSteps=options.baseSteps??baseSteps,T=options.totalTime??totalTime,errorStrength=options.errorStrength??1;
    if(!Number.isInteger(nominalSteps)||nominalSteps<8||nominalSteps>400||!Number.isFinite(T)||T<=0||!Number.isFinite(errorStrength)||errorStrength<0||errorStrength>1)throw new RangeError('invalid comparison settings');
    const steps=Math.max(2,Math.round(nominalSteps/alpha)),h=T/steps;
    const schedule=Array.from({length:steps+1},(_,k)=>sigmaMax*Math.exp(Math.log(sigmaMin/sigmaMax)*k/steps));
    schedule[0]=sigmaMax;schedule[steps]=sigmaMin;
    const configs=schedule.map(s=>G.prepare({d,centers,sigmas:[s]}));
    function run(kind){
      const points=[sample(oldPoint,0,kind,lambda,errorStrength,schedule,configs,h)];
      // Every transition, INCLUDING the first, uses this same time step and
      // score-plus-guidance rule. We retain all N+1 clock records even after
      // the sample reaches the manifold or the illustrative path wanders off.
      for(let k=0;k<steps;k++){
        const before=points[k],next=before.point.map((v,i)=>v+h*before.velocity[i]);
        if(!next.every(Number.isFinite))throw new RangeError('non-finite illustrated trajectory');
        const after=sample(next,k+1,kind,lambda,errorStrength,schedule,configs,h);
        after.stepSize=h;after.effectiveStepSize=h/before.normalizer;
        after.effectiveStepSizes=before.normalizers.map(v=>h/v);
        after.priorDisplacement=before.prior.map((v,i)=>h*v/before.normalizers[i]);
        after.guidanceDisplacement=before.task.map((v,i)=>h*v/before.normalizers[i]);
        after.errorDisplacement=before.error.map((v,i)=>h*v/before.normalizers[i]);
        after.deltaObjective=after.objective-before.objective;points.push(after);
      }
      const last=points.at(-1),failed=points.some(p=>p.errorActivation>1e-5);
      const status=last.constraintSatisfied&&lambda>0?'In constrained manifold region':last.onManifold?'On manifold':failed?'Off-manifold / illustrative mismatch':'Still approaching the manifold';
      return{kind,points,status,steps,onManifold:last.onManifold,nearData:last.onManifold,
        inConstrainedRegion:last.constraintSatisfied,constraintSatisfied:last.constraintSatisfied,illustrativeFailure:failed};
    }
    const conditional=run('conditional'),unconditioned=run('unconditioned');
    return{d,lambda,alpha,manifold:{type:'circle',center:[0,0],radius:R},constraint:{type:'rightArc',minX,interiorX},
      onManifoldTolerance:manifoldTolerance,trainingBandFactor:bandFactor,noiseScales:nScales,
      initialPoint:oldPoint.slice(),oldPoint:oldPoint.slice(),steps,stepSize:h,totalTime:T,baseSteps:nominalSteps,
      sigmaMin,sigmaMax,schedule,conditional,unconditioned,taskRetained:true,errorStrength,transverseDamping,
      note:'One continuous timeline with a prescribed decreasing noise schedule on the left and position-selected noise information on the right. Every update uses the same guided rule and time step within its run. Off-band learned-score failure is conceptual, not a measured neural-network result. Blue training bands are schematic.'};
  }
  const presets=Object.freeze({baseline:{lambda:0,alpha:1},mild:{lambda:1,alpha:1},strong:{lambda:9,alpha:1},coarse:{lambda:2,alpha:4}});
  return Object.freeze({guidedComparison,scaleWeights,geometry,penalty,taskAt,taskCurvature,guidanceDamping,stateOnlyTransverseDamping,presets,
    constants:{d,sigmaMin,sigmaMax,nScales,baseSteps,totalTime,radius:R,minX,interiorX,transverseDamping,trainingBandFactor:bandFactor,onManifoldTolerance:manifoldTolerance,initialPoint:oldPoint}});
});
