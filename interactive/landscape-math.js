/* Exact Gaussian-mixture mathematics for the interactive research essay.
 * No dependencies; safe to load from file:// and in a browser worker.
 *
 * API (window.GaussianLandscapeMath, or require(...) in Node):
 *   prepare({d=2,sigmaMin=.22,sigmaMax=1.3,nScales=32,centers=[[0,0]],sigmas?})
 *     -> immutable cached configuration. Explicit sigmas override the schedule.
 *   evaluate(x,y,prepared)
 *     -> {logDensity,score:[sx,sy],beta,barycenter:[mx,my],
 *         centerResponsibilities:[...],isFinite}
 *   step([x,y],prepared,eta,mode='raw'|'normalized')
 *     -> {point,before,after,deltaLogDensity,isFinite}
 *   geometricScales(min,max,n), logGamma(z), radialMass(r,sigma,d),
 *   logRadialMass(r,sigma,d), ambientRelative(r,sigma), shellSummary(sigma,d),
 *   singleCenterStability(prepared), singleCenterRadialMass(r,prepared).
 *
 * The displayed x/y coordinates lie in an invariant two-dimensional slice:
 * all centers and the other d-2 sample coordinates are zero off this plane.
 * Evaluated height is the FULL d-dimensional ambient density on that slice,
 * not a 2D marginal or a radial probability density. The exact target is
 * p_MN=(M sum_i sigma_i^2)^-1 sum_j,i sigma_i^2 N_d(x;mu_j,sigma_i^2 I).
 * No trajectory clipping or monotonicity correction is performed here.
 */
(function attach(root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.GaussianLandscapeMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function makeAPI() {
  'use strict';
  const LOG_TWO_PI = Math.log(2 * Math.PI);
  const LANCZOS = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
  ];

  function finitePositive(value, name) {
    if (!Number.isFinite(value) || value <= 0) throw new RangeError(name + ' must be finite and positive');
    return value;
  }

  function dimension(value) {
    if (!Number.isInteger(value) || value < 2) throw new RangeError('d must be an integer at least 2');
    return value;
  }

  function logGamma(z) {
    finitePositive(z, 'logGamma argument');
    if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    z -= 1;
    let x = LANCZOS[0];
    for (let i = 1; i < LANCZOS.length; i++) x += LANCZOS[i] / (z + i);
    const t = z + 7.5;
    return 0.5 * LOG_TWO_PI + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  function geometricScales(min, max, count) {
    finitePositive(min, 'sigmaMin'); finitePositive(max, 'sigmaMax');
    if (max < min) throw new RangeError('sigmaMax must be at least sigmaMin');
    if (!Number.isInteger(count) || count < 1 || count > 4096) throw new RangeError('nScales must be an integer from 1 to 4096');
    if (count === 1) return [min];
    const width = Math.log(max / min);
    const result = Array.from({length:count}, (_, i) => min * Math.exp(width * i / (count - 1)));
    result[0] = min; result[result.length - 1] = max;
    return result;
  }

  function prepare(input) {
    input = input || {};
    const d = dimension(input.d === undefined ? 2 : input.d);
    const sourceCenters = input.centers === undefined ? [[0, 0]] : input.centers;
    if (!Array.isArray(sourceCenters) || !sourceCenters.length) throw new RangeError('at least one center is required');
    const centers = sourceCenters.map((c) => {
      const x = Array.isArray(c) ? c[0] : c.x;
      const y = Array.isArray(c) ? c[1] : c.y;
      if (!Number.isFinite(x) || !Number.isFinite(y)) throw new RangeError('each center needs finite x and y coordinates');
      return Object.freeze([x, y]);
    });
    const sigmas = input.sigmas === undefined
      ? geometricScales(input.sigmaMin === undefined ? 0.22 : input.sigmaMin,
          input.sigmaMax === undefined ? 1.3 : input.sigmaMax,
          input.nScales === undefined ? 32 : input.nScales)
      : Array.from(input.sigmas);
    if (!sigmas.length) throw new RangeError('at least one noise scale is required');
    sigmas.forEach(s => finitePositive(s, 'sigma'));
    const sumVariance = sigmas.reduce((sum, s) => sum + s * s, 0);
    if (!Number.isFinite(sumVariance)) throw new RangeError('noise variances overflow');
    const invVariances = sigmas.map(s => 1 / (s * s));
    if (invVariances.some(v => !Number.isFinite(v))) throw new RangeError('noise precision overflows');
    return Object.freeze({
      d,
      centers: Object.freeze(centers),
      sigmas: Object.freeze(sigmas),
      sigmaMin: Math.min.apply(null, sigmas),
      sigmaMax: Math.max.apply(null, sigmas),
      nScales: sigmas.length,
      invVariances: Object.freeze(invVariances),
      logCoefficients: Object.freeze(sigmas.map(s => (2 - d) * Math.log(s))),
      logNormalizer: Math.log(centers.length * sumVariance) + 0.5 * d * LOG_TWO_PI,
      sumVariance,
      _gaussianLandscapePrepared: true
    });
  }

  function invalidEvaluation(count) {
    return {logDensity:-Infinity, score:[NaN,NaN], beta:NaN,
      barycenter:[NaN,NaN], centerResponsibilities:Array(count).fill(0), isFinite:false};
  }

  function evaluate(x, y, config) {
    if (!config || !config._gaussianLandscapePrepared) throw new TypeError('evaluate requires a prepare(...) configuration');
    const centers = config.centers, n = centers.length;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return invalidEvaluation(n);
    const distances = new Float64Array(n);
    let maximum = -Infinity;
    for (let j = 0; j < n; j++) {
      const dx = x - centers[j][0], dy = y - centers[j][1];
      const r2 = dx * dx + dy * dy;
      distances[j] = r2;
      for (let i = 0; i < config.nScales; i++) {
        const term = config.logCoefficients[i] - 0.5 * r2 * config.invVariances[i];
        if (term > maximum) maximum = term;
      }
    }
    if (!Number.isFinite(maximum)) return invalidEvaluation(n);
    let sum = 0, precisionSum = 0, baryX = 0, baryY = 0;
    const centerResponsibilities = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      let centerMass = 0, centerPrecision = 0;
      for (let i = 0; i < config.nScales; i++) {
        const term = config.logCoefficients[i] - 0.5 * distances[j] * config.invVariances[i];
        const relativeWeight = Math.exp(term - maximum);
        centerMass += relativeWeight;
        centerPrecision += relativeWeight * config.invVariances[i];
      }
      centerResponsibilities[j] = centerMass;
      sum += centerMass;
      precisionSum += centerPrecision;
      baryX += centerPrecision * centers[j][0];
      baryY += centerPrecision * centers[j][1];
    }
    const beta = precisionSum / sum;
    baryX /= precisionSum; baryY /= precisionSum;
    const score = [beta * (baryX - x), beta * (baryY - y)];
    const logDensity = maximum + Math.log(sum) - config.logNormalizer;
    for (let j = 0; j < n; j++) centerResponsibilities[j] /= sum;
    const isFinite = Number.isFinite(logDensity) && Number.isFinite(beta)
      && Number.isFinite(baryX) && Number.isFinite(baryY)
      && Number.isFinite(score[0]) && Number.isFinite(score[1]);
    return {logDensity, score, beta, barycenter:[baryX,baryY], centerResponsibilities, isFinite};
  }

  function step(point, config, eta, mode) {
    if (!Number.isFinite(eta) || eta < 0) throw new RangeError('eta must be finite and nonnegative');
    mode = mode || 'raw';
    if (mode !== 'raw' && mode !== 'normalized') throw new RangeError('mode must be raw or normalized');
    const before = evaluate(point[0], point[1], config);
    const direction = mode === 'normalized'
      ? [before.barycenter[0] - point[0], before.barycenter[1] - point[1]]
      : before.score;
    const next = [point[0] + eta * direction[0], point[1] + eta * direction[1]];
    const after = evaluate(next[0], next[1], config);
    const deltaLogDensity = after.logDensity - before.logDensity;
    return {point:next, before, after, deltaLogDensity,
      isFinite:before.isFinite && after.isFinite && Number.isFinite(deltaLogDensity)};
  }

  function logRadialMass(r, sigma, d) {
    finitePositive(sigma, 'sigma'); dimension(d);
    if (!Number.isFinite(r) || r < 0) return -Infinity;
    if (r === 0) return -Infinity; // d>=2: the radial PDF is zero at the origin.
    return (d - 1) * Math.log(r) - 0.5 * (r / sigma) * (r / sigma)
      - (0.5 * d - 1) * Math.log(2) - logGamma(0.5 * d) - d * Math.log(sigma);
  }

  function radialMass(r, sigma, d) {
    return Math.exp(logRadialMass(r, sigma, d));
  }

  function ambientRelative(r, sigma) {
    finitePositive(sigma, 'sigma');
    if (!Number.isFinite(r) || r < 0) return 0;
    return Math.exp(-0.5 * (r / sigma) * (r / sigma));
  }

  function shellSummary(sigma, d) {
    finitePositive(sigma, 'sigma'); dimension(d);
    const mean = sigma * Math.sqrt(2) * Math.exp(logGamma(0.5 * (d + 1)) - logGamma(0.5 * d));
    const variance = Math.max(0, d * sigma * sigma - mean * mean);
    return {
      d, sigma,
      mode:sigma * Math.sqrt(d - 1), mean,
      radiusApprox:sigma * Math.sqrt(d), standardDeviation:Math.sqrt(variance),
      relativeWidth:Math.sqrt(variance) / mean,
      log2VolumeDoubling:d, effectiveMassDoubling:4,
      log2WeightedDensityDoubling:2 - d
    };
  }

  function singleCenterStability(config) {
    if (config.centers.length !== 1) throw new RangeError('the exact isotropic center threshold requires one center');
    const center = config.centers[0], ev = evaluate(center[0], center[1], config);
    return {betaAtCenter:ev.beta, localRawEulerLimit:2 / ev.beta,
      conservativeRawEulerLimit:2 * config.sigmaMin * config.sigmaMin,
      normalizedEulerLimit:2,
      note:'Strict inequalities; the raw limit is local, at the single-center peak.'};
  }

  function singleCenterRadialMass(r, config) {
    if (config.centers.length !== 1) throw new RangeError('radial symmetry requires one center');
    let result = 0;
    for (const sigma of config.sigmas) result += sigma * sigma * radialMass(r, sigma, config.d);
    return result / config.sumVariance;
  }

  return Object.freeze({prepare, evaluate, step, geometricScales, logGamma,
    logRadialMass, radialMass, ambientRelative, shellSummary,
    singleCenterStability, singleCenterRadialMass});
});
