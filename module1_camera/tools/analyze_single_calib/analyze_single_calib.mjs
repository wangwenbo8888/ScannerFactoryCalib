#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, basename } from 'node:path';

const TOOL_VERSION = '1.1.0';
const SCHEMA = 'factory_calib.single_calib_summary.v1';
const SEMVER = '1.1.0';
const FP_VERSION = 'v2-single';
const ESTIMATOR = 'v1';
const IQR_K = 1.5, SIGMA_K = 3, TOP_N = 5;
const FLOOR_MIN = 0.02, FLOOR_K = 3, FLOOR_P90_K = 2;
const REPROJ_TH_DEFAULT = 0.05, ROW_ALIGN_TH_DEFAULT = 0.05, STEREO_TH_DEFAULT = 0.07, STEREO_RMS_TH_DEFAULT = 0.30;
const PASS_FRAC = 0.8, SHAPE_FLOOR_RATIO = 0.85;
const IMBAL_HI = 1.5, IMBAL_LO = 0.67, IMBAL_PCT_TH = 30, IMBAL_SIDE_CONSIST = 0.8;
const CORNER_EMPTY_MAX = 2, TILT_MIN_BUCKET_N = 5, SUBPIX_P90_TH = 0.5;
const R_BETA_GAP = 0.3, R_RATIO_TH = 5, R_RATIO_CORR_TH = 0.3, R_CORRLR_TH = 0.4;
const RADIUS_EDGES = [0, 0.1, 0.2, 0.3, 0.4, 0.7];
const GUARD = { R1: 20, R2: 15, R3: 20, R4: 15, R5: 15, R6: 15, R7: 20, R8: 0 };
const CONCESSION_ROUNDS_DEFAULT = 3;
const PF_COLS = ['frame_id', 'orig_index', 'left_file', 'right_file', 'valid', 'skip_reason',
  'corner_count', 'mean_subpix_delta_px', 'reproj_L_px', 'reproj_R_px', 'reproj_ratio',
  'epi_raw_px', 'row_align_px', 'stereo_reproj_px', 'extrin_deg', 'extrin_mm',
  'dist_L_mm', 'dist_R_mm', 'tilt_deg', 'cover_pct', 'board_center_u_px', 'board_center_v_px',
  'region_L', 'region_R', 'temp_c', 'exposure_us'];
const PC_COLS = ['frame_id', 'corner_idx', 'grid_row', 'grid_col', 'x_L_px', 'y_L_px', 'x_R_px', 'y_R_px',
  'resid_dx_L_px', 'resid_dy_L_px', 'resid_dx_R_px', 'resid_dy_R_px', 'epi_raw_px', 'row_align_dy_px', 'radius_norm'];
const BETACF_MAX_IT = 200, BETACF_EPS = 3e-12, BETACF_STOP = 3e-7, IBETA_BTMAX = 1e300;
const SIG_P01 = 0.01, SIG_P05 = 0.05, SIG_P10 = 0.10;

function parseArgs(argv) {
  const a = { out: null, reprojTh: null, rowAlignTh: null, stereoTh: null, stereoRmsTh: null, concessionRounds: CONCESSION_ROUNDS_DEFAULT };
  const pos = [];
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === '--out') a.out = argv[++i];
    else if (t === '--reproj-th') a.reprojTh = parseFloat(argv[++i]);
    else if (t === '--row-align-th') a.rowAlignTh = parseFloat(argv[++i]);
    else if (t === '--stereo-th') a.stereoTh = parseFloat(argv[++i]);
    else if (t === '--stereo-rms-th') a.stereoRmsTh = parseFloat(argv[++i]);
    else if (t === '--concession-rounds') a.concessionRounds = parseInt(argv[++i], 10);
    else pos.push(t);
  }
  if (!pos.length) throw new Error('usage: node analyze_single_calib.mjs <dataset_dir> [--out dir] [--reproj-th px] [--row-align-th px] [--stereo-th px] [--stereo-rms-th px] [--concession-rounds n]');
  a.dir = pos[0];
  return a;
}

function splitCsvLine(line) {
  const out = [];
  let cur = '', q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else q = false; }
      else cur += c;
    } else if (c === '"') q = true;
    else if (c === ',') { out.push(cur); cur = ''; }
    else cur += c;
  }
  out.push(cur);
  return out;
}

function readCsv(path, requiredCols) {
  const text = readFileSync(path, 'utf8');
  const lines = text.split(/\r?\n/).filter((l) => l.length && !l.startsWith('#'));
  if (lines.length < 2) throw new Error(`${basename(path)}: 空文件`);
  const hdr = splitCsvLine(lines[0]);
  const missing = requiredCols.filter((c) => !hdr.includes(c));
  if (missing.length) throw new Error(`${basename(path)}: 缺列 ${missing.join('/')}`);
  const idx = Object.fromEntries(hdr.map((h, i) => [h, i]));
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const f = splitCsvLine(lines[i]);
    const o = {};
    for (const c of requiredCols) o[c] = f[idx[c]] ?? '';
    rows.push(o);
  }
  return rows;
}

const num = (s) => (s === '' || s === undefined || s === null) ? null : Number(s);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const pstd = (a) => Math.sqrt(a.reduce((s, v) => s + (v - mean(a)) ** 2, 0) / a.length);
function quantile(sorted, q) {
  const n = sorted.length;
  if (!n) return null;
  const pos = q * (n - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function fiveNum(a) {
  const v = [...a].sort((x, y) => x - y), n = v.length;
  const m = mean(v);
  const s2 = v.reduce((s, x) => s + (x - m) ** 2, 0) / n;
  const s3 = v.reduce((s, x) => s + (x - m) ** 3, 0) / n;
  const s4 = v.reduce((s, x) => s + (x - m) ** 4, 0) / n;
  return { n, mean: m, p50: quantile(v, 0.5), p90: quantile(v, 0.9), max: v[n - 1],
    std: Math.sqrt(s2), skew: s2 > 0 ? s3 / s2 ** 1.5 : null, kurt: s2 > 0 ? s4 / s2 ** 2 - 3 : null,
    estimator: ESTIMATOR, subset: 'valid' };
}
function pearson(xs, ys) {
  const p = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i++)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) p.push([xs[i], ys[i]]);
  if (p.length < 3) return null;
  const mx = mean(p.map((q) => q[0])), my = mean(p.map((q) => q[1]));
  let sxy = 0, sxx = 0, syy = 0;
  for (const [x, y] of p) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2; }
  if (sxx === 0 || syy === 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}
function gammaln(z) {
  const G = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - gammaln(1 - z);
  z -= 1;
  let x = G[0];
  for (let i = 1; i < G.length; i++) x += G[i] / (z + i);
  const t = z + G.length - 1.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function betacf(a, b, x) {
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap;
  if (Math.abs(d) < BETACF_EPS) d = BETACF_EPS;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= BETACF_MAX_IT; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < BETACF_EPS) d = BETACF_EPS;
    c = 1 + aa / c; if (Math.abs(c) < BETACF_EPS) c = BETACF_EPS;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < BETACF_EPS) d = BETACF_EPS;
    c = 1 + aa / c; if (Math.abs(c) < BETACF_EPS) c = BETACF_EPS;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < BETACF_STOP) return h;
  }
  return h;
}
function ibeta(a, b, x) {
  if (!(a > 0) || !(b > 0) || !Number.isFinite(x)) return null;
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lnbt = gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x);
  if (!Number.isFinite(lnbt)) return null;
  const bt = lnbt > Math.log(IBETA_BTMAX) ? IBETA_BTMAX : Math.exp(lnbt);
  if (bt === 0) return x < (a + 1) / (a + b + 2) ? 0 : 1;
  if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
  return 1 - bt * betacf(b, a, 1 - x) / b;
}
function corrPvalue(r, n) {
  if (r == null || !Number.isFinite(r) || n == null || !Number.isFinite(n) || n < 3) return null;
  if (Math.abs(r) >= 1) return 0;
  const df = n - 2;
  const t2 = r * r * df / (1 - r * r);
  return ibeta(df / 2, 0.5, df / (t2 + df));
}
function sigMark(r, n) {
  const p = corrPvalue(r, n);
  if (p == null) return '';
  if (p < SIG_P01) return '**';
  if (p < SIG_P05) return '*';
  if (p < SIG_P10) return '.';
  return '';
}
function corrEntry(xs, ys) {
  const r = pearson(xs, ys);
  let n = 0;
  for (let i = 0; i < Math.min(xs.length, ys.length); i++)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) n++;
  if (r == null) return { r: null, p: null, sig: '', n };
  return { r, p: corrPvalue(r, n), sig: sigMark(r, n), n };
}
function logLogSlope(ys, xs) {
  const p = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i++)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i]) && xs[i] > 0 && ys[i] > 0)
      p.push([Math.log(xs[i]), Math.log(ys[i])]);
  const slopeOf = (pts) => {
    if (pts.length < 2) return null;
    const mx = mean(pts.map((q) => q[0])), my = mean(pts.map((q) => q[1]));
    let sxy = 0, sxx = 0;
    for (const q of pts) { sxy += (q[0] - mx) * (q[1] - my); sxx += (q[0] - mx) ** 2; }
    return sxx > 1e-9 ? sxy / sxx : null;
  };
  if (p.length < 2) return { beta: null, betaRobust: null, leverage: null };
  const beta = slopeOf(p);
  if (beta == null || p.length === 2) return { beta, betaRobust: null, leverage: null };
  const mx = mean(p.map((q) => q[0]));
  let wi = 0, wd = -1;
  for (let i = 0; i < p.length; i++) { const d = Math.abs(p[i][0] - mx); if (d > wd) { wd = d; wi = i; } }
  const betaRobust = slopeOf(p.filter((_, i) => i !== wi));
  if (betaRobust == null) return { beta, betaRobust: null, leverage: null };
  return { beta, betaRobust, leverage: Math.abs(beta - betaRobust) };
}

function pngSize(path) {
  const b = readFileSync(path);
  if (b.length < 24 || b.toString('ascii', 1, 4) !== 'PNG') return null;
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

function configFingerprint(path) {
  if (!existsSync(path)) return null;
  const j = JSON.parse(readFileSync(path, 'utf8'));
  const canon = (o) => {
    if (Array.isArray(o)) return o.map(canon);
    if (o && typeof o === 'object') return Object.keys(o).sort().map((k) => [k, canon(o[k])]);
    return o;
  };
  return createHash('sha256').update(JSON.stringify(canon(j))).digest('hex').slice(0, 16);
}

function binIndex(v, edges) {
  for (let i = edges.length - 1; i >= 0; i--) if (v >= edges[i]) return i;
  return 0;
}

function svgWrap(w, h, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><rect width="${w}" height="${h}" fill="white"/>${body}</svg>\n`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const ds = args.dir;
  const problems = [];
  const jsonPath = join(ds, 'camera_calib.json');
  const pfPath = join(ds, 'camera_calib_perframe.csv');
  const pcPath = join(ds, 'camera_calib_percorner.csv');
  for (const [p, n] of [[jsonPath, 'camera_calib.json'], [pfPath, 'camera_calib_perframe.csv'], [pcPath, 'camera_calib_percorner.csv']])
    if (!existsSync(p)) problems.push(`缺少 ${n}`);
  let calib = null;
  if (!problems.length) {
    try { calib = JSON.parse(readFileSync(jsonPath, 'utf8')); }
    catch (e) { problems.push(`camera_calib.json 解析失败: ${e.message}`); }
  }
  if (problems.length) {
    console.error('无法判定：');
    for (const p of problems) console.error('  - ' + p);
    process.exit(2);
  }
  const pf = readCsv(pfPath, PF_COLS);
  const pc = readCsv(pcPath, PC_COLS);

  const validF = pf.filter((r) => r.valid === '1');
  const invalidF = pf.filter((r) => r.valid !== '1');
  const W = calib.imageSize?.[0] ?? null, H = calib.imageSize?.[1] ?? null;
  const byCol = (c) => validF.map((r) => num(r[c]));
  const errMax = validF.map((r) => Math.max(num(r.reproj_L_px), num(r.reproj_R_px)));
  const rowAlign = byCol('row_align_px');
  const stereo = byCol('stereo_reproj_px');
  let reprojL = byCol('reproj_L_px');
  let reprojR = byCol('reproj_R_px');
  const jpvL = calib.intrinsic?.left?.per_view_errors;
  const jpvR = calib.intrinsic?.right?.per_view_errors;
  if (Array.isArray(jpvL) && jpvL.length === validF.length && Array.isArray(jpvR) && jpvR.length === validF.length) {
    reprojL = jpvL;
    reprojR = jpvR;
  }
  const distL = byCol('dist_L_mm');
  const distR = byCol('dist_R_mm');
  const tilt = byCol('tilt_deg');

  const skipByReason = invalidF.reduce((m, r) => { const k = r.skip_reason || 'unknown'; m[k] = (m[k] || 0) + 1; return m; }, {});

  let measured = null, pngUnverified = false;
  try {
    const ldir = join(ds, 'left');
    if (existsSync(ldir)) {
      const fs2 = readdirSync(ldir).filter((f) => f.toLowerCase().endsWith('.png')).sort();
      if (fs2.length) measured = pngSize(join(ldir, fs2[0]));
      else pngUnverified = true;
    } else pngUnverified = true;
  } catch { pngUnverified = true; }

  const nineGrid = {};
  for (const r of validF) nineGrid[r.region_L] = (nineGrid[r.region_L] || 0) + 1;
  const cornerCells = ['TL', 'TR', 'BL', 'BR'];
  const emptyCornerCells = cornerCells.filter((k) => !nineGrid[k]);
  const distSorted = [...distL].sort((a, b) => a - b);
  const distQuantiles = [0.1, 0.25, 0.5, 0.75, 0.9].map((q) => quantile(distSorted, q));
  const tiltBuckets = { frontal: 0, oblique: 0, steep: 0 };
  for (const t of tilt) { if (t < 15) tiltBuckets.frontal++; else if (t <= 45) tiltBuckets.oblique++; else tiltBuckets.steep++; }
  const tiltOrientations = Object.values(tiltBuckets).filter((n) => n >= TILT_MIN_BUCKET_N).length;
  const subpix = fiveNum(validF.map((r) => num(r.mean_subpix_delta_px)));

  let percornerCountOk = true;
  const expectCorners = validF.reduce((s, r) => s + num(r.corner_count), 0);
  if (pc.length !== expectCorners) percornerCountOk = false;
  let frameIdOk = true;
  validF.forEach((r, i) => { if (num(r.frame_id) !== i + 1) frameIdOk = false; });

  const imageMismatch = measured != null && W != null && (measured[0] !== W || measured[1] !== H);
  const auditFail = imageMismatch || !percornerCountOk || !frameIdOk;

  const dyAll = pc.map((r) => num(r.row_align_dy_px));
  const dySorted = [...dyAll].sort((a, b) => a - b);
  const dyMed = quantile(dySorted, 0.5);
  const mad = quantile([...dyAll].map((v) => Math.abs(v - dyMed)).sort((a, b) => a - b), 0.5);
  const sigmaFloor = 1.4826 * mad;
  const rowAlignTh = args.rowAlignTh ?? ROW_ALIGN_TH_DEFAULT;
  const rowAlignSrc = args.rowAlignTh != null ? 'cli' : 'default';

  const raFive = fiveNum(rowAlign);
  const errMean = mean(errMax);
  const reprojCombinedMean = mean([...reprojL, ...reprojR]);
  const gates = [];
  const g1v = errMean, g1t = args.reprojTh ?? REPROJ_TH_DEFAULT;
  gates.push({ id: 'G1', name: 'reproj', stat: 'mean(max L,R)', value: g1v, threshold: g1t, source: args.reprojTh != null ? 'cli' : 'default',
    grade: g1v > g1t ? 'FAIL' : (g1v > PASS_FRAC * g1t ? 'WARN' : 'PASS'), note: `json 合并均值 ${reprojCombinedMean.toFixed(6)}` });
  const g2v = raFive.p50, g2t = rowAlignTh;
  const g2grade = g2v > g2t ? 'FAIL' : ((raFive.p90 > FLOOR_P90_K * g2t || g2v > PASS_FRAC * g2t) ? 'WARN' : 'PASS');
  gates.push({ id: 'G2', name: 'row_align', stat: 'P50', value: g2v, threshold: g2t, source: rowAlignSrc,
    grade: g2grade, note: `noiseFloor σ=${sigmaFloor.toFixed(4)} / P90=${raFive.p90.toFixed(4)} / 双条件 P90≤${(FLOOR_P90_K * g2t).toFixed(4)}` });
  const stFive = fiveNum(stereo);
  const g3v = stFive.p90, g3t = args.stereoTh ?? STEREO_TH_DEFAULT;
  gates.push({ id: 'G3', name: 'stereo_reproj', stat: 'P90', value: g3v, threshold: g3t, source: args.stereoTh != null ? 'cli' : 'default',
    grade: g3v > g3t ? 'FAIL' : (g3v > PASS_FRAC * g3t ? 'WARN' : 'PASS'), note: '' });
  const stereoRms = calib.extrinsic?.stereoReprojError ?? null;
  const g4t = args.stereoRmsTh ?? STEREO_RMS_TH_DEFAULT;
  if (stereoRms != null)
    gates.push({ id: 'G4', name: 'stereo_rms', stat: 'value', value: stereoRms, threshold: g4t, source: args.stereoRmsTh != null ? 'cli' : 'default',
      grade: stereoRms > g4t ? 'FAIL' : (stereoRms > PASS_FRAC * g4t ? 'WARN' : 'PASS'), note: 'cv::stereoCalibrate 返回值（含 R/T 拟合，平方放大口径）' });

  const dist = {
    reprojL: fiveNum(reprojL), reprojR: fiveNum(reprojR), rowAlign: raFive,
    stereoReproj: stFive, extrinDeg: fiveNum(byCol('extrin_deg')), extrinMm: fiveNum(byCol('extrin_mm')),
    reprojCombinedMean,
  };
  const shapeRatio = raFive.mean > 0 ? raFive.p50 / raFive.mean : null;
  const shapeRatioReproj = dist.reprojL.mean > 0 ? dist.reprojL.p50 / dist.reprojL.mean : null;
  const shapeVerdict = shapeRatio != null && shapeRatio >= SHAPE_FLOOR_RATIO ? 'floor' : 'tail';

  const vals = [...errMax];
  const frameRefs = validF.map((r, i) => ({ view: i + 1, file: r.left_file, v: errMax[i] }));
  const sortedV = [...vals].sort((a, b) => a - b);
  const q1 = quantile(sortedV, 0.25), q3 = quantile(sortedV, 0.75), iqr = q3 - q1;
  const med = quantile(sortedV, 0.5), sd = pstd(vals);
  const iqrOut = frameRefs.filter((f) => f.v < q1 - IQR_K * iqr || f.v > q3 + IQR_K * iqr).map((f) => f.view);
  const ms3Out = frameRefs.filter((f) => Math.abs(f.v - med) > SIGMA_K * sd).map((f) => f.view);
  const topIdx = [...frameRefs].sort((a, b) => b.v - a.v).slice(0, TOP_N).map((f) => f.view);
  const cnt = new Map();
  for (const i of [...iqrOut, ...ms3Out, ...topIdx]) cnt.set(i, (cnt.get(i) || 0) + 1);
  const stableViews = [...cnt.entries()].filter((e) => e[1] >= 2).map((e) => e[0]).sort((a, b) => a - b);
  const refByView = Object.fromEntries(frameRefs.map((f) => [f.view, f]));
  const stableOut = stableViews.map((v) => {
    const r = validF[v - 1];
    return { file: r.left_file, view: v, reprojL: num(r.reproj_L_px), reprojR: num(r.reproj_R_px), rowAlign: num(r.row_align_px) };
  });

  const ratioImb = validF.map((r) => num(r.reproj_ratio));
  let leftWorse = 0, rightWorse = 0;
  for (const q of ratioImb) { if (q > IMBAL_HI) leftWorse++; else if (q < IMBAL_LO) rightWorse++; }
  const imbPct = ((leftWorse + rightWorse) / validF.length) * 100;
  const sideConsist = (leftWorse + rightWorse) > 0 ? Math.max(leftWorse, rightWorse) / (leftWorse + rightWorse) : null;
  const corrLR = corrEntry(reprojL, reprojR);
  const frameIdx = validF.map((r) => num(r.orig_index));
  const tempC = validF.map((r) => num(r.temp_c));
  const expo = validF.map((r) => num(r.exposure_us));
  const sidecarPresent = tempC.some((v) => v != null) || expo.some((v) => v != null);
  const mkCorr = (xs, label) => {
    if (!xs.some((v) => v != null)) return { r: null, note: label === 'temp' || label === 'exposure' ? 'sidecar absent' : 'data missing' };
    return corrEntry(xs, errMax);
  };
  const correlations = {
    dist: mkCorr(distL, 'dist'), tilt: mkCorr(tilt, 'tilt'),
    coverInv: mkCorr(byCol('cover_pct').map((v) => (v == null ? null : -v)), 'coverInv'),
    frameIdx: mkCorr(frameIdx, 'frameIdx'),
    temp: sidecarPresent ? mkCorr(tempC, 'temp') : { r: null, note: 'sidecar absent' },
    exposure: sidecarPresent ? mkCorr(expo, 'exposure') : { r: null, note: 'sidecar absent' },
  };
  const betaL = logLogSlope(reprojL, distL);
  const betaR = logLogSlope(reprojR, distR);
  const corrReprojAlign = corrEntry(errMax, rowAlign);
  const innerOuterRatio = errMean > 0 ? raFive.p50 / errMean : null;

  const residBins = [], alignBins = [];
  for (let b = 0; b < RADIUS_EDGES.length - 1; b++) { residBins.push({ bin: b, n: 0, s: 0 }); alignBins.push({ bin: b, n: 0, s: 0 }); }
  for (const r of pc) {
    const rn = num(r.radius_norm);
    if (rn == null) continue;
    const b = Math.min(binIndex(rn, RADIUS_EDGES), RADIUS_EDGES.length - 2);
    const magL = Math.hypot(num(r.resid_dx_L_px), num(r.resid_dy_L_px));
    const magR = Math.hypot(num(r.resid_dx_R_px), num(r.resid_dy_R_px));
    residBins[b].n++; residBins[b].s += (magL + magR) / 2;
    alignBins[b].n++; alignBins[b].s += Math.abs(num(r.row_align_dy_px));
  }
  const residualByRadius = { edges: RADIUS_EDGES, estimator: ESTIMATOR,
    meanResidMag: residBins.map((b) => (b.n ? b.s / b.n : null)), n: residBins.map((b) => b.n) };
  const rowAlignByRadius = { edges: RADIUS_EDGES, estimator: ESTIMATOR,
    meanAbsDy: alignBins.map((b) => (b.n ? b.s / b.n : null)), n: alignBins.map((b) => b.n) };

  let levFile = null, levVal = -1;
  validF.forEach((r) => { const v = num(r.extrin_mm); if (v > levVal) { levVal = v; levFile = r.left_file; } });

  const farTh = distQuantiles[3];
  const farCovers = validF.filter((r) => num(r.dist_L_mm) >= farTh).map((r) => num(r.cover_pct));
  const farCoverMean = farCovers.length ? mean(farCovers) : null;

  const fp = [];
  const pushFp = (id, name, n, evidence, judge) => {
    const g = GUARD[id];
    let verdict;
    if (g > 0 && n < g) verdict = 'insufficient';
    else verdict = judge(evidence);
    fp.push({ id, name, verdict, n, guard: g, evidence });
  };
  const sigOk = (e) => e.sig === '*' || e.sig === '**';
  pushFp('R1', '双目对焦/距离响应不一致', validF.length, { betaL: betaL.beta, betaR: betaR.beta },
    (e) => e.betaL != null && e.betaR != null && e.betaL * e.betaR < 0 && Math.abs(e.betaL - e.betaR) > R_BETA_GAP ? 'hit' : 'clear');
  pushFp('R2', '姿态覆盖不足', validF.length, { tiltOrientations, emptyCornerCells: emptyCornerCells.length, nineGrid },
    (e) => (e.tiltOrientations < 2 || e.emptyCornerCells > CORNER_EMPTY_MAX) ? 'hit' : 'clear');
  pushFp('R3', '远距小板亚像素不足', validF.length, { corrDist: correlations.dist, farCoverMean },
    (e) => (sigOk(e.corrDist) && e.farCoverMean != null && e.farCoverMean < 15) ? 'hit' : 'clear');
  pushFp('R4', '外参嫌疑', validF.length, { innerOuterRatio, corrReprojAlign },
    (e) => (e.innerOuterRatio != null && e.innerOuterRatio > R_RATIO_TH && e.corrReprojAlign.r != null && Math.abs(e.corrReprojAlign.r) < R_RATIO_CORR_TH) ? 'hit' : 'clear');
  pushFp('R5', '单相机劣化', validF.length, { imbPct, leftWorse, rightWorse, sideConsist },
    (e) => (e.imbPct > IMBAL_PCT_TH && e.sideConsist != null && e.sideConsist >= IMBAL_SIDE_CONSIST) ? 'hit' : 'clear');
  pushFp('R6', '全局场景问题', validF.length, { corrLR },
    (e) => (e.corrLR.r != null && e.corrLR.r > R_CORRLR_TH && e.corrLR.sig === '**') ? 'hit' : 'clear');
  pushFp('R7', '采集期漂移', validF.length, { corrFrameIdx: correlations.frameIdx, corrTemp: correlations.temp },
    (e) => (sigOk(e.corrFrameIdx) || (e.corrTemp.r != null && sigOk(e.corrTemp))) ? 'hit' : 'clear');
  pushFp('R8', '数据量/姿态病态', validF.length, { validFrames: validF.length, tiltOrientations, subpixP90: subpix.p90 },
    (e) => (e.validFrames < 10 || e.tiltOrientations < 2 || (e.subpixP90 != null && e.subpixP90 > SUBPIX_P90_TH)) ? 'hit' : 'clear');

  const actions = [];
  if (imageMismatch) actions.push({ code: 'RECHECK_CONFIG', detail: `image_size 配置 ${W}x${H} ≠ 实测 ${measured[0]}x${measured[1]}` });
  if (!percornerCountOk) actions.push({ code: 'RECHECK_CONFIG', detail: `percorner 行数 ${pc.length} ≠ Σcorner_count ${expectCorners}` });
  if (!frameIdOk) actions.push({ code: 'RECHECK_CONFIG', detail: 'perframe frame_id 序列不连续' });
  if (stableOut.length) actions.push({ code: 'RERUN_FRAMES', targets: stableOut.map((o) => o.file), detail: `${stableOut.length} 个稳定离群帧` });
  const gapCells = emptyCornerCells.slice();
  if (tiltOrientations < 2) gapCells.push('斜朝向');
  if (gapCells.length) actions.push({ code: 'ADD_POSE', targets: gapCells, detail: '覆盖缺口' });
  const physMap = { R1: '双目对焦/光圈一致性', R5: '镜头清洁/单相机检查', R6: '漫射均匀光照、棋盘清洁', R7: '采集期温度稳定性' };
  const phys = fp.filter((f) => f.verdict === 'hit' && physMap[f.id]).map((f) => `${f.id} ${physMap[f.id]}`);
  if (phys.length) actions.push({ code: 'PHYSICAL_CHECK', targets: phys });
  const anyFail = gates.some((g) => g.grade === 'FAIL');
  const anyWarn = gates.some((g) => g.grade === 'WARN');
  if (!anyFail && anyWarn) actions.push({ code: 'CONCESSION', detail: '仅 WARN 级，可申请让步放行（记录判定依据）' });
  actions.push({ code: 'ESCALATE', detail: `连续 ${args.concessionRounds} 轮未过→转工程师` });
  const lamp = auditFail ? 'FAIL' : (anyFail ? 'FAIL' : (anyWarn ? 'WARN' : 'PASS'));
  const arbitration = auditFail ? 'audit' : (anyFail ? 'gate' : (anyWarn ? 'gate-warn' : 'none'));
  const lampIcon = { PASS: '🟢', WARN: '🟡', FAIL: '🔴' }[lamp];

  const outDir = args.out ?? join(ds, 'analysis_single');
  mkdirSync(outDir, { recursive: true });
  mkdirSync(join(outDir, 'charts'), { recursive: true });
  const generated = new Date().toLocaleString('sv-SE').replace('T', ' ');
  const datasetName = basename(ds.replace(/[\\/]+$/, ''));

  const terminal = [];
  const g4 = gates.find((g) => g.id === 'G4');
  terminal.push(`| lamp | verdict | reproj | row_align(P50/σ_floor/T) | stereo(P90) | stereoRMS(v/T) | frames |`);
  terminal.push('|---|---|---|---|---|---|---|');
  terminal.push(`| ${lampIcon} | ${lamp}(${arbitration}) | ${g1v.toFixed(4)}/${g1t} | ${g2v.toFixed(4)}/σ${sigmaFloor.toFixed(4)}/T${g2t.toFixed(4)}(${rowAlignSrc}) | ${g3v.toFixed(4)}/${g3t} | ${g4 ? `${g4.value.toFixed(4)}/${g4.threshold}` : 'absent'} | ${validF.length}/${pf.length} |`);
  terminal.push(`指纹: ${fp.filter((f) => f.verdict === 'hit').map((f) => f.id + '⚠').join(' ') || '无命中'} | 形态: ${shapeVerdict} | 离群: ${stableOut.map((o) => o.file).join(',') || '无'}`);
  for (const a of actions) terminal.push(`动作 ${a.code}: ${a.targets ? a.targets.join(',') + ' ' : ''}${a.detail ?? ''}`);

  const L = [];
  L.push(`# 单次标定体检 · ${datasetName}　【${lampIcon} ${lamp}】`);
  L.push('');
  L.push(`生成 ${generated}｜帧 ${validF.length}/${pf.length}（skip ${invalidF.length}）｜参考温 ${calib.referenceTemp} °C｜工具 v${TOOL_VERSION}｜指纹口径 ${FP_VERSION}`);
  L.push('');
  L.push(`## 判定仲裁（优先级 审计＞门禁＞解稳＞哨兵）`);
  L.push('');
  L.push(`- 仲裁结果：**${arbitration}**` + (imageMismatch ? `——image_size 配置 ${W}x${H} ≠ 实测 ${measured[0]}x${measured[1]}` : ''));
  if (!percornerCountOk) L.push(`- percorner 行数 ${pc.length} ≠ Σcorner_count ${expectCorners}`);
  if (!frameIdOk) L.push('- perframe frame_id 序列不连续');
  L.push('');
  L.push('## 门禁表');
  L.push('');
  L.push('| 门禁 | 指标 | 值 | 阈值 | 来源 | 到阈距离 | 判定 | 备注 |');
  L.push('|---|---|---|---|---|---|---|---|');
  for (const g of gates)
    L.push(`| ${g.id} ${g.name} | ${g.stat} | ${g.value.toFixed(4)} | ${g.threshold.toFixed(4)} | ${g.source} | ${(g.value - g.threshold).toFixed(4)} | ${g.grade} | ${g.note} |`);
  L.push('');
  L.push('> epi_raw 仅作对照列不设门禁（原始像素×F 口径含 ~0.3px 伪影基线）；G2 阈值来源可审计。');
  L.push('');
  L.push('## 建议动作');
  L.push('');
  for (const a of actions)
    L.push(`- **${a.code}**${a.targets ? `: ${a.targets.join(', ')}` : ''}${a.detail ? `——${a.detail}` : ''}`);
  L.push('');
  L.push('<details><summary>一、输入审计</summary>');
  L.push('');
  L.push(`- image_size：声明 ${W}x${H} / 实测 ${measured ? measured.join('x') : (pngUnverified ? 'unverified（无 PNG 可读）' : '未检测')} ${imageMismatch ? '❌ 不符' : '✓'}`);
  L.push(`- 帧利用：${validF.length} 有效 / ${invalidF.length} 跳过；skip 分级 ${JSON.stringify(skipByReason)}`);
  L.push(`- 覆盖：九宫格 ${JSON.stringify(nineGrid)}；四角空格 ${emptyCornerCells.length}${emptyCornerCells.length ? '（' + emptyCornerCells.join(',') + '）' : ''}；tilt 朝向 ${tiltOrientations} 桶 ${JSON.stringify(tiltBuckets)}`);
  L.push(`- 距离分位 P10/25/50/75/90：${distQuantiles.map((q) => q.toFixed(0)).join('/')} mm`);
  L.push(`- 检测质量 subpix：mean ${subpix.mean.toFixed(3)} / P90 ${subpix.p90.toFixed(3)} / max ${subpix.max.toFixed(3)} px`);
  L.push(`- 数据完整性：frame_id 连续 ${frameIdOk ? '✓' : '✗'}；percorner 行数 ${pc.length}${percornerCountOk ? '✓' : '✗'}`);
  L.push(`- 侧车：${sidecarPresent ? '在' : 'absent（温度/曝光相关不可测）'}`);
  L.push('');
  L.push('</details>');
  L.push('');
  L.push('<details><summary>二、分布与离群</summary>');
  L.push('');
  L.push('| 序列 | n | mean | P50 | P90 | max | std | 偏度 | 峰度 |');
  L.push('|---|---|---|---|---|---|---|---|---|');
  for (const [k, d] of Object.entries({ reprojL: dist.reprojL, reprojR: dist.reprojR, rowAlign: dist.rowAlign, stereoReproj: dist.stereoReproj, extrinDeg: dist.extrinDeg, extrinMm: dist.extrinMm }))
    L.push(`| ${k} | ${d.n} | ${d.mean.toFixed(4)} | ${d.p50.toFixed(4)} | ${d.p90.toFixed(4)} | ${d.max.toFixed(4)} | ${d.std.toFixed(4)} | ${d.skew?.toFixed(2) ?? 'N/A'} | ${d.kurt?.toFixed(2) ?? 'N/A'} |`);
  L.push('');
  L.push(`- 形态判定：**${shapeVerdict}**（row_align P50/mean=${shapeRatio?.toFixed(3)}；reprojL=${shapeRatioReproj?.toFixed(3)}，阈 ${SHAPE_FLOOR_RATIO}）——${shapeVerdict === 'floor' ? '系统性地板，删帧无效，查系统原因' : '尾部型，删帧/重拍有效'}`);
  L.push(`- 离群三法：IQR[${iqrOut.join(',')}] medSigma3[${ms3Out.join(',')}] TOP5[${topIdx.join(',')}]；稳定离群（≥2 法）${stableOut.length} 帧`);
  if (stableOut.length) {
    L.push('');
    L.push('| 文件 | view | reprojL | reprojR | row_align |');
    L.push('|---|---|---|---|---|');
    for (const o of stableOut) L.push(`| ${o.file} | ${o.view} | ${o.reprojL.toFixed(4)} | ${o.reprojR.toFixed(4)} | ${o.rowAlign.toFixed(4)} |`);
  }
  L.push('');
  L.push('</details>');
  L.push('');
  L.push('<details><summary>三、归因诊断</summary>');
  L.push('');
  L.push(`- 内外参分诊：row_align P50 / reproj mean = ${innerOuterRatio?.toFixed(3) ?? 'N/A'}（<2 内参侧 / 2~5 混合 / >${R_RATIO_TH} 外参侧）`);
  L.push(`- 左右：corrLR=${corrLR.r?.toFixed(4) ?? 'N/A'}${corrLR.sig}；失衡帧 ${imbPct.toFixed(1)}%（左差 ${leftWorse}/右差 ${rightWorse}，侧一致度 ${(sideConsist * 100 || 0).toFixed(0)}%）`);
  L.push(`- 相关性（误差~X，帧级）：dist ${fmtCorr(correlations.dist)}｜tilt ${fmtCorr(correlations.tilt)}｜cover⁻ ${fmtCorr(correlations.coverInv)}｜帧序 ${fmtCorr(correlations.frameIdx)}｜温度 ${fmtCorr(correlations.temp)}｜曝光 ${fmtCorr(correlations.exposure)}`);
  L.push(`- β 距离斜率（log-log）：βL=${betaL.beta?.toFixed(3) ?? 'N/A'}(robust ${betaL.betaRobust?.toFixed(3) ?? '-'}) / βR=${betaR.beta?.toFixed(3) ?? 'N/A'}(robust ${betaR.betaRobust?.toFixed(3) ?? '-'})`);
  L.push(`- 残差按半径桶均值：${residualByRadius.meanResidMag.map((v) => v?.toFixed(4) ?? '-').join(' | ')}`);
  L.push(`- row_align|dy| 按半径桶均值：${rowAlignByRadius.meanAbsDy.map((v) => v?.toFixed(4) ?? '-').join(' | ')}`);
  L.push(`- 外参自洽最大杠杆帧：${levFile ?? 'N/A'}（Δmm ${levVal.toFixed(3)}）`);
  L.push('');
  L.push('</details>');
  L.push('');
  L.push('<details><summary>四、指纹全表（口径 ' + FP_VERSION + '）</summary>');
  L.push('');
  L.push('| 规则 | 判定 | n/守卫 | 证据 |');
  L.push('|---|---|---|---|');
  for (const f of fp)
    L.push(`| ${f.id} ${f.name} | ${f.verdict === 'hit' ? '⚠hit' : f.verdict} | ${f.n}/${f.guard} | ${JSON.stringify(f.evidence).slice(0, 120)} |`);
  L.push('');
  L.push('</details>');
  L.push('');
  L.push('<details><summary>五、附录·指标口径与版本</summary>');
  L.push('');
  L.push(`- schema ${SCHEMA} semver ${SEMVER}｜estimator ${ESTIMATOR}｜工具 v${TOOL_VERSION}｜指纹 ${FP_VERSION}`);
  L.push(`- G2 阈值：默认 ${ROW_ALIGN_TH_DEFAULT}（CLI 可覆盖）；噪声地板审计 σ_floor=1.4826×MAD(row_align_dy)=${sigmaFloor.toFixed(4)}，3σ 参考线 ${(FLOOR_K * sigmaFloor).toFixed(4)}；PASS 双条件 P50≤T 且 P90≤${FLOOR_P90_K}T`);
  L.push(`- G4 立体 RMS：extrinsic.stereoReprojError（cv::stereoCalibrate 返回值，含 R/T 拟合与平方放大口径，与 G3 逐帧 stereo_reproj 不同源）；默认 ${STEREO_RMS_TH_DEFAULT}（--stereo-rms-th 可覆盖）；字段缺席时跳过该门禁不影响灯号`);
  L.push('- epi_raw 对照口径：原始像素×F（含伪影基线，不判门禁）；row_align：矫正域（正确口径）');
  L.push(`- config 指纹：${configFingerprint(join(ds, 'config.json')) ?? 'absent'}`);
  L.push('');
  L.push('</details>');
  L.push('');
  L.push('![perframe](charts/perframe.svg)');
  L.push('');
  L.push('![radius](charts/radius_binned.svg)');
  L.push('');
  L.push('![field](charts/residual_field.svg)');
  L.push('');

  const summary = {
    schema: SCHEMA, semver: SEMVER, toolVersion: TOOL_VERSION, fingerprint_logic_version: FP_VERSION,
    meta: {
      dataset: datasetName, generated, frames: { valid: validF.length, input: pf.length }, skipByReason,
      configFingerprint: configFingerprint(join(ds, 'config.json')),
      imageSize: { declared: [W, H], measured, sidecar: sidecarPresent ? 'present' : 'absent' },
    },
    verdict: {
      lamp, arbitration,
      gates: gates.map((g) => ({ id: g.id, metric: g.name, stat: g.stat, value: round6(g.value), threshold: round6(g.threshold), source: g.source, distance: round6(g.value - g.threshold), grade: g.grade, noiseFloor: g.id === 'G2' ? round6(sigmaFloor) : undefined })),
      actions: actions.map((a) => ({ code: a.code, targets: a.targets ?? undefined, detail: a.detail ?? undefined })),
    },
    audit: {
      imageMismatch, frameIdOk, percornerCountOk, percornerRows: pc.length,
      coverage: { nineGrid, emptyCornerCells, distQuantiles: distQuantiles.map(round6), tiltOrientations, tiltBuckets, poseBinKey: { schema: 'grid3x3-dist4-tilt3' } },
      subpix: { mean: round6(subpix.mean), p90: round6(subpix.p90), max: round6(subpix.max) },
    },
    distribution: { ...Object.fromEntries(Object.entries(dist).map(([k, v]) => [k, v && v.n !== undefined ? v : round6(v)])), shapeVerdict, shapeRatio: round6(shapeRatio) },
    outliers: {
      stable: stableOut, methods: { iqr: iqrOut, medSigma3: ms3Out, top5: topIdx },
    },
    attribution: {
      innerOuterRatio: round6(innerOuterRatio), imbalancePct: round6(imbPct), leftWorse, rightWorse, corrLR,
      corrReprojAlign, correlations,
      betaL, betaR,
      residualByRadius, rowAlignByRadius,
      leverageFrame: levFile, leverageMm: round6(levVal),
    },
    fingerprints: fp,
    frames: { ref: 'camera_calib_perframe.csv', key: ['frame_id', 'orig_index', 'left_file'] },
    calibRef: { json: 'camera_calib.json', schema: calib.schema },
  };

  writeSvgCharts(outDir, errMax, frameRefs, g1t, residBins, alignBins, pc, W, H);

  let ok = true;
  try { writeFileSync(join(outDir, 'report.md'), L.join('\n'), 'utf8'); } catch (e) { ok = false; console.error('report.md 写失败: ' + e.message); }
  try { writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 2), 'utf8'); } catch (e) { ok = false; console.error('summary.json 写失败: ' + e.message); }
  console.log(terminal.join('\n'));
  console.log(`report: ${join(outDir, 'report.md')}`);
  console.log(`summary: ${join(outDir, 'summary.json')}`);
  if (!ok) process.exit(1);
}

function fmtCorr(c) {
  if (c.r == null) return `N/A（${c.note ?? 'missing'}）`;
  return `${c.r.toFixed(4)}${c.sig}（p=${c.p == null ? '?' : c.p < 1e-4 ? c.p.toExponential(1) : c.p.toFixed(4)}, n=${c.n}）`;
}
function round6(v) { return typeof v === 'number' && Number.isFinite(v) ? Math.round(v * 1e6) / 1e6 : v; }

function writeSvgCharts(outDir, errMax, frameRefs, th, residBins, alignBins, pc, W, H) {
  const bw = 1400, bh = 360, pad = 40;
  const maxV = Math.max(...errMax, th * 1.3);
  const n = errMax.length;
  const barW = Math.max((bw - 2 * pad) / n - 0.4, 0.5);
  let b = '';
  errMax.forEach((v, i) => {
    const h = (v / maxV) * (bh - 2 * pad);
    const stable = frameRefs[i] && false;
    b += `<rect x="${(pad + i * (bw - 2 * pad) / n).toFixed(1)}" y="${(bh - pad - h).toFixed(1)}" width="${barW.toFixed(2)}" height="${h.toFixed(1)}" fill="${v > th ? '#d62728' : '#1f77b4'}"/>`;
  });
  const yTh = bh - pad - (th / maxV) * (bh - 2 * pad);
  b += `<line x1="${pad}" y1="${yTh}" x2="${bw - pad}" y2="${yTh}" stroke="#d62728" stroke-width="1.5"/><text x="${bw - pad}" y="${yTh - 5}" text-anchor="end" font-size="12" fill="#d62728">reproj 门禁 ${th}</text>`;
  const yW = bh - pad - (PASS_FRAC * th / maxV) * (bh - 2 * pad);
  b += `<line x1="${pad}" y1="${yW}" x2="${bw - pad}" y2="${yW}" stroke="#ff7f0e" stroke-dasharray="4 3"/><text x="${bw - pad}" y="${yW - 5}" text-anchor="end" font-size="12" fill="#ff7f0e">WARN ${PASS_FRAC * th}</text>`;
  writeFileSync(join(outDir, 'charts', 'perframe.svg'), svgWrap(bw, bh, b), 'utf8');

  const bw2 = 900, bh2 = 320, pad2 = 60;
  const nb = residBins.length;
  const max2 = Math.max(...residBins.map((x) => x.n ? x.s / x.n : 0), ...alignBins.map((x) => x.n ? x.s / x.n : 0), 1e-6);
  let b2 = '';
  for (let i = 0; i < nb; i++) {
    const x0 = pad2 + i * (bw2 - 2 * pad2) / nb;
    const w = (bw2 - 2 * pad2) / nb;
    const hr = residBins[i].n ? (residBins[i].s / residBins[i].n) / max2 * (bh2 - 2 * pad2) : 0;
    const ha = alignBins[i].n ? (alignBins[i].s / alignBins[i].n) / max2 * (bh2 - 2 * pad2) : 0;
    b2 += `<rect x="${x0 + 4}" y="${bh2 - pad2 - hr}" width="${w / 2 - 8}" height="${hr}" fill="#1f77b4"/>`;
    b2 += `<rect x="${x0 + w / 2 + 4}" y="${bh2 - pad2 - ha}" width="${w / 2 - 8}" height="${ha}" fill="#2ca02c"/>`;
    b2 += `<text x="${x0 + w / 2}" y="${bh2 - pad2 + 16}" text-anchor="middle" font-size="11">${RADIUS_EDGES[i]}-${RADIUS_EDGES[i + 1]}</text>`;
  }
  b2 += `<text x="${pad2}" y="${pad2 - 20}" font-size="12" fill="#1f77b4">■ 残差幅值</text><text x="${pad2 + 90}" y="${pad2 - 20}" font-size="12" fill="#2ca02c">■ |row_align_dy|</text>`;
  writeFileSync(join(outDir, 'charts', 'radius_binned.svg'), svgWrap(bw2, bh2, b2), 'utf8');

  const bw3 = 560, bh3 = 440, pad3 = 40;
  let b3 = `<line x1="${pad3}" y1="${pad3}" x2="${bw3 - pad3}" y2="${pad3}" stroke="#999"/><line x1="${pad3}" y1="${bh3 - pad3}" x2="${bw3 - pad3}" y2="${bh3 - pad3}" stroke="#999"/><line x1="${pad3}" y1="${pad3}" x2="${pad3}" y2="${bh3 - pad3}" stroke="#999"/><line x1="${bw3 - pad3}" y1="${pad3}" x2="${bw3 - pad3}" y2="${bh3 - pad3}" stroke="#999"/>`;
  const cells = 9;
  const acc = Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => ({ n: 0, sx: 0, sy: 0 })));
  for (const r of pc) {
    const x = num(r.x_L_px), y = num(r.y_L_px);
    if (x == null || y == null || !W || !H) continue;
    const ci = x < W / 3 ? 0 : x < 2 * W / 3 ? 1 : 2;
    const ri = y < H / 3 ? 0 : y < 2 * H / 3 ? 1 : 2;
    acc[ri][ci].n++; acc[ri][ci].sx += num(r.resid_dx_L_px) || 0; acc[ri][ci].sy += num(r.resid_dy_L_px) || 0;
  }
  let maxMag = 1e-9;
  for (let ri = 0; ri < 3; ri++) for (let ci = 0; ci < 3; ci++) if (acc[ri][ci].n) maxMag = Math.max(maxMag, Math.hypot(acc[ri][ci].sx / acc[ri][ci].n, acc[ri][ci].sy / acc[ri][ci].n));
  for (let ri = 0; ri < 3; ri++) for (let ci = 0; ci < 3; ci++) {
    const cx = pad3 + (ci + 0.5) * (bw3 - 2 * pad3) / 3;
    const cy = pad3 + (ri + 0.5) * (bh3 - 2 * pad3) / 3;
    const c = acc[ri][ci];
    if (!c.n) { b3 += `<text x="${cx}" y="${cy}" text-anchor="middle" font-size="12" fill="#aaa">n=0</text>`; continue; }
    const dx = c.sx / c.n, dy = c.sy / c.n;
    const k = 40 / maxMag;
    b3 += `<line x1="${cx}" y1="${cy}" x2="${cx + dx * k}" y2="${cy + dy * k}" stroke="#d62728" stroke-width="2" marker-end="url(#arr)"/>`;
    b3 += `<text x="${cx + 4}" y="${cy - 6}" font-size="10" fill="#333">(${dx.toFixed(3)},${dy.toFixed(3)})×${c.n}</text>`;
  }
  b3 = `<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#d62728"/></marker></defs>` + b3;
  writeFileSync(join(outDir, 'charts', 'residual_field.svg'), svgWrap(bw3, bh3, b3), 'utf8');
}

main();
