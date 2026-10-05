#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve, basename, join } from 'node:path';
import { pathToFileURL } from 'node:url';

const IQR_K = 1.5, SIGMA_K = 3, TOP_N = 5, BOTH_BAD_K = 1.5, COVERAGE_BINS = 8, EPS = 1e-12;
const BETACF_MAX_IT = 200, BETACF_EPS = 3e-12, BETACF_STOP = 3e-7, IBETA_BTMAX = 300;
const SIG_P01 = 0.01, SIG_P05 = 0.05, SIG_P10 = 0.10, IMBALANCE_HI = 1.5, IMBALANCE_LO = 0.67;
const MIN_VALID_FRAMES = 10, ORIENT_ANGLE_DEG = 15, CORNER_EMPTY_MAX = 2;
const GRID_NX = 17, GRID_NY = 13, GRID_RANGE = 0.9, GRID_MARGIN_PX = 60, LOGLOG_EPS = 1e-9;
const UNDISTORT_MAX_IT = 20, UNDISTORT_EPS = 1e-12, LEVERAGE_WARN = 0.15;
const REF_TYPICAL_REPROJ = 0.1, REF_REPROJ_SOURCE = 'BoofCV 经验值';
const FP_BETA_GAP = 0.3, FP_CORR_STRONG = 0.4, FP_CORR_SIGSTAR = 1, FP_BETA_PHYS_LO = -1, FP_BETA_PHYS_HI = 0.05;
const FP_EPI_RATIO = 5, FP_EPI_DECOUPLE = 0.3, FP_IMBAL_PCT = 30, FP_DOMINANCE = 2, FP_DRIFT = 0.5;
const FP_MIN_FRAMES = 10, FP_LR_STRONG = 0.4;

const GLOSSARY = [
  {
    group: '门禁与总评',
    rows: [
      ['verdict', '全部门禁通过=PASS / 有观察项=WARN / 有未过项=FAIL'],
      ['N/A', '该项不可测或数据缺失（如无棋盘 config 时几何类指标）'],
      ['reproj_mean', '平均重投影误差(px)——标定参数把棋盘角点「投影回」图像与实测角点的平均偏差，越小内参越准'],
      ['epipolar_mean', '平均极线误差(px)——左图角点经双目位姿换算后偏离右图极线的平均距离，检验外参（双目相对位姿）'],
      ['stereoRms', '双目整体重投影 RMS(px)，内参＋外参联合误差'],
      ['skipped_pct', '角点提取失败被跳过的帧占比'],
      ['基线 \\|T\\|', '双目两相机光心的物理间距（mm），跨批应稳定'],
      ['table:*', '温度补偿表存在性与档数检查（应为 61 档）'],
    ],
  },
  {
    group: '逐帧分布',
    rows: [
      ['reprojL / reprojR', '每帧左/右相机的重投影误差'],
      ['P50 / P90', '一半 / 90% 的帧低于该值，P90 更能反映最差的那批帧'],
      ['std', '帧间波动，越大越不整齐'],
    ],
  },
  {
    group: '一致性',
    rows: [
      ['焦距Δ%', '左右焦距相对差，同型号镜头应接近'],
      ['主点偏离', '光心离图像中心的像素距离，过大提示装配或拟合问题'],
      ['validRoi 覆盖', '立体矫正后有效画面占比，越大越好'],
    ],
  },
  {
    group: '离群帧',
    rows: [
      ['view#', '跳帧后重排的有效帧序号（1 起），不等于原始文件编号'],
      ['IQR / medSigma3 / TOP5', '三种找坏帧方法（箱线 / 3σ / 最差5帧），≥2 法命中=稳定离群'],
      ['距离mm', '该帧棋盘到相机的拍摄距离'],
      ['斜角°', '棋盘平面与光轴夹角，越斜越难测准'],
      ['覆盖%', '棋盘占画面面积比，太小则角点像素不足'],
      ['L-R比', '左右误差之比，偏离 1 表示单侧劣化'],
      ['双差', '重投影与极线误差都大=该帧本身质量差'],
    ],
  },
  {
    group: '相关性',
    rows: [
      ['corrDist / corrAngle / corrCoverInv', '误差与距离/斜角/覆盖的关系强度（-1~1），绝对值越大越相关'],
      ['星号', '显著性——* p<0.05，** p<0.01，. p<0.10（弱证据），无星=可能只是巧合'],
      ['偏相关', '剔除干扰因素后的净关系（如去掉距离影响的纯边缘效应）'],
      ['corrFrameIdx', '误差随拍摄顺序的趋势，强相关提示采集期漂移（如热漂）'],
      ['偏度/峰度', '分布形态——偏度>0 有坏帧长尾；峰度高=厚尾，极端误差帧偏多'],
      ['fxFyPct', '焦距纵横比差，应接近 0'],
    ],
  },
  {
    group: '标定可信度',
    rows: [
      ['帧数≥10', 'OpenCV 实践门槛，太少解不稳'],
      ['姿态方向', '棋盘至少两个朝向，单一朝向方程病态'],
      ['宫格角落', '棋盘须拍到画面边角，否则边缘畸变欠约束'],
      ['参考量级', '良好制作下典型重投影误差约 0.1px（BoofCV 经验值）；门禁阈值仍以 config 为准'],
    ],
  },
  {
    group: '双目一致性',
    rows: [
      ['L-R差值 std', '同帧左右误差差的波动，大=两相机表现不一致'],
      ['corrLR', '左右误差同帧相关，高=坏帧是全局的（光照/棋盘问题）'],
      ['corrReprojEpi', '内参×外参误差相关，≈0 说明两者不同源'],
      ['β', '距离-误差斜率——≈-1 纯板尺寸效应 / ≈0 像素噪声主导 / >0 越远越糊（对焦可疑）；robust=剔除杠杆帧后值；⚠杠杆敏感=结果受个别帧影响大'],
      ['失衡帧', '左右误差比 >1.5 或 <0.67 的帧'],
    ],
  },
  {
    group: '批次间投影差异',
    rows: [
      ['maxPx / meanPx', '两批参数对同一画面点投影位置的最大/平均偏差'],
      ['kOnlyMaxPx', '只换 K 不换畸变的偏差——分辨差异来自 K 还是畸变'],
      ['一致性指示器', '只说明「两批参数差多少」，不能区分镜头变化/拟合波动/参数漂移'],
      ['Δ vs 首列', '与第一个输入批次相比的百分比变化（首列自身为 N/A）'],
      ['高误差 view 重现', '两批按 view# 对齐后，同名帧在两批都超过各自 P90——重现提示系统性问题'],
      ['Δfx%/Δcx/Δcy/Δk1~k3/Δp1/Δp2', '两批参数逐项增量（焦距%/主点/径向/切向畸变系数）'],
    ],
  },
  {
    group: '指纹',
    rows: [
      ['R1~R8', '八类常见问题特征判定（对焦不一致/覆盖不足/远距小板/外参嫌疑/单相机劣化/全局场景/热漂/数据病态）'],
      ['判定标记', '⚠命中 / —排除 / ? 证据不足；「初步重现」=2-3 批线索级，≥4 批结论级'],
    ],
  },
];

function parseArgs(argv) {
  const p = { inputs: [], out: null, reprojTh: null, epipolarTh: null, noCharts: false, errors: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') {
      const v = argv[++i];
      if (v === undefined) p.errors.push('--out requires a value');
      else p.out = v;
    } else if (a === '--reproj-th') {
      const v = Number(argv[++i]);
      if (!Number.isFinite(v)) p.errors.push(`--reproj-th requires a finite number (got ${argv[i] ?? 'nothing'})`);
      else p.reprojTh = v;
    } else if (a === '--epipolar-th') {
      const v = Number(argv[++i]);
      if (!Number.isFinite(v)) p.errors.push(`--epipolar-th requires a finite number (got ${argv[i] ?? 'nothing'})`);
      else p.epipolarTh = v;
    } else if (a === '--no-charts') p.noCharts = true;
    else if (a.startsWith('--')) p.errors.push(`unknown option ${a}`);
    else p.inputs.push(a);
  }
  if (p.inputs.length === 0) p.errors.push('no input json');
  return p;
}

function loadRun(path, usedLabels) {
  const label = basename(dirname(resolve(path)));
  let uniq = label, k = 2;
  while (usedLabels.has(uniq)) uniq = `${label}#${k++}`;
  usedLabels.add(uniq);
  let raw;
  try { raw = JSON.parse(readFileSync(path, 'utf8')); }
  catch (e) { return { path, label: uniq, fatal: String(e.message || e) }; }
  return { path, label: uniq, raw };
}

function stamp() {
  const d = new Date(), z = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}_${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
}
function defaultOutDir() {
  const here = dirname(process.argv[1]);
  return resolve(here, 'out', `camera_calib_${stamp()}`);
}

const num = v => typeof v === 'number' && Number.isFinite(v) ? v : null;
const arr = v => Array.isArray(v) ? v.filter(x => typeof x === 'number' && Number.isFinite(x)) : [];

function quantiles(xs, q) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}
function stats(xs) {
  if (!xs.length) return null;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const varr = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
  return { n: xs.length, min: Math.min(...xs), max: Math.max(...xs), mean, std: Math.sqrt(varr),
    p50: quantiles(xs, .5), p90: quantiles(xs, .9) };
}
function mat3(m) {
  return m && m.length === 3 && m.every(r => Array.isArray(r) && r.length === 3 && r.every(x => typeof x === 'number' && Number.isFinite(x))) ? m : null;
}
function matCol(v) {
  return v && v.length === 3 && v.every(r => { const x = Array.isArray(r) ? r[0] : r; return typeof x === 'number' && Number.isFinite(x); })
    ? v.map(r => Array.isArray(r) ? r[0] : r) : null;
}
function vecNorm(v) { return Math.hypot(...v); }
function vec3(v) {
  if (!Array.isArray(v) || v.length !== 3) return null;
  const out = [];
  for (const r of v) {
    const x = Array.isArray(r) ? r[0] : r;
    if (typeof x !== 'number' || !Number.isFinite(x)) return null;
    out.push(x);
  }
  return out;
}
function rodrigues(r) {
  const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const th = Math.hypot(...r);
  if (th < EPS) return I;
  const x = r[0] / th, y = r[1] / th, z = r[2] / th, s = Math.sin(th), c = 1 - Math.cos(th);
  const K = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  return I.map((row, i) => row.map((v, j) => v + s * K[i][j] + c * (K[i][0] * K[0][j] + K[i][1] * K[1][j] + K[i][2] * K[2][j])));
}
function boardGeometry(m, cfgBoard) {
  if (!cfgBoard) return null;
  const W = num(m.imageSize?.[0]), H = num(m.imageSize?.[1]);
  if (!W || !H) return null;
  const hw = (cfgBoard.width - 1) * cfgBoard.square / 2, hh = (cfgBoard.height - 1) * cfgBoard.square / 2;
  const corners = [[-hw, -hh, 0], [hw, -hh, 0], [hw, hh, 0], [-hw, hh, 0]];
  const geo = {};
  for (const s of ['left', 'right']) {
    const sd = m.sides?.[s], K = sd?.K;
    if (!K) { geo[s] = null; continue; }
    const pv = sd.perView, rv = sd.rvecs, tv = sd.tvecs;
    const n = Math.min(pv.length, rv.length, tv.length);
    const views = [];
    for (let i = 0; i < n; i++) {
      const r = vec3(rv[i]), t = vec3(tv[i]);
      if (!r || !t) { views.push(null); continue; }
      const R = rodrigues(r);
      const pts = corners.map(cn => {
        const X = R[0][0] * cn[0] + R[0][1] * cn[1] + t[0];
        const Y = R[1][0] * cn[0] + R[1][1] * cn[1] + t[1];
        const Z = R[2][0] * cn[0] + R[2][1] * cn[1] + t[2];
        if (Z <= 0) return null;
        const u = K[0][0] * X / Z + K[0][2], v = K[1][1] * Y / Z + K[1][2];
        return Number.isFinite(u) && Number.isFinite(v) ? [u, v] : null;
      });
      if (pts.some(p => p === null)) { views.push(null); continue; }
      const us = pts.map(p => p[0]), vs = pts.map(p => p[1]);
      const u0 = Math.min(...us), u1 = Math.max(...us), v0 = Math.min(...vs), v1 = Math.max(...vs);
      views.push({
        dist: Math.hypot(...t),
        angleDeg: Math.acos(Math.max(-1, Math.min(1, R[2][2]))) * 180 / Math.PI,
        coverPct: (u1 - u0) * (v1 - v0) / (W * H) * 100,
        centerOff: Math.hypot((u0 + u1) / 2 - W / 2, (v0 + v1) / 2 - H / 2),
        uc: (u0 + u1) / 2,
        vc: (v0 + v1) / 2,
      });
    }
    geo[s] = { perView: views };
  }
  return geo;
}
function detectOutliers(xs) {
  const empty = { methods: { iqr: [], medSigma3: [], top5: [] }, stable: [] };
  const pairs = [];
  for (let i = 0; i < xs.length; i++) if (Number.isFinite(xs[i])) pairs.push([i, xs[i]]);
  if (!pairs.length) return empty;
  const vals = pairs.map(p => p[1]);
  const q1 = quantiles(vals, .25), q3 = quantiles(vals, .75), iqr = q3 - q1;
  const med = quantiles(vals, .5);
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
  const iqrOut = pairs.filter(p => p[1] < q1 - IQR_K * iqr || p[1] > q3 + IQR_K * iqr).map(p => p[0]);
  const medSigma3 = pairs.filter(p => Math.abs(p[1] - med) > SIGMA_K * sd).map(p => p[0]);
  const top5 = [...pairs].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, TOP_N).map(p => p[0]);
  const cnt = new Map();
  for (const i of [...iqrOut, ...medSigma3, ...top5]) cnt.set(i, (cnt.get(i) ?? 0) + 1);
  const stable = [...cnt.entries()].filter(e => e[1] >= 2).map(e => e[0]).sort((a, b) => a - b);
  return { methods: { iqr: iqrOut, medSigma3, top5 }, stable };
}
function viewMaxErr(r) {
  const a = r.reprojL, b = r.reprojR;
  return a != null && b != null ? Math.max(a, b) : a ?? b;
}
function analyzeOutliers(m, geo) {
  const lv = m.sides?.left?.perView ?? [], rv = m.sides?.right?.perView ?? [];
  const epiArr = m.extrinsic?.epiPerView ?? [], epiMean = m.extrinsic?.epiMean;
  const n = Math.min(lv.length, rv.length);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const gl = geo?.left?.perView?.[i] ?? null, gr = geo?.right?.perView?.[i] ?? null;
    rows.push({
      view: i, reprojL: lv[i] ?? null, reprojR: rv[i] ?? null, epi: epiArr[i] ?? null,
      distL: gl ? gl.dist : null, distR: gr ? gr.dist : null,
      angleL: gl ? gl.angleDeg : null, angleR: gr ? gr.angleDeg : null,
      coverL: gl ? gl.coverPct : null, coverR: gr ? gr.coverPct : null,
      centerOffL: gl ? gl.centerOff : null, centerOffR: gr ? gr.centerOff : null,
    });
  }
  const top5 = [...rows].sort((a, b) => (viewMaxErr(b) ?? -Infinity) - (viewMaxErr(a) ?? -Infinity) || a.view - b.view).slice(0, TOP_N)
    .map(r => ({
      ...r, maxErr: viewMaxErr(r),
      lrRatio: r.reprojL != null && r.reprojR > 0 ? r.reprojL / r.reprojR : null,
      bothBad: r.epi != null && epiMean != null ? r.epi > BOTH_BAD_K * epiMean : null,
    }));
  const outliers = { reproj: detectOutliers(rows.map(viewMaxErr)), epipolar: detectOutliers(rows.map(r => r.epi)) };
  return { rows, outliers, top5 };
}
function loadBoardCfg(run) {
  const dir = dirname(resolve(run.path));
  for (const c of [join(dir, 'config.json'), join(dirname(dir), 'config.json')]) {
    try {
      const cfg = JSON.parse(readFileSync(c, 'utf8'));
      const w = num(cfg?.chessboard?.width), h = num(cfg?.chessboard?.height), sq = num(cfg?.chessboard?.square_size_mm);
      if (w != null && h != null && sq != null) return { width: w, height: h, square: sq };
    } catch {}
  }
  return null;
}
function pearson(xs, ys) {
  const p = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i++)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) p.push([xs[i], ys[i]]);
  if (p.length < 3) return null;
  const mx = p.reduce((a, b) => a + b[0], 0) / p.length, my = p.reduce((a, b) => a + b[1], 0) / p.length;
  let sxy = 0, sxx = 0, syy = 0;
  for (const [x, y] of p) { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; syy += (y - my) ** 2; }
  if (sxx === 0 || syy === 0) return null;
  return sxy / Math.sqrt(sxx * syy);
}
function bins(xs, k) {
  const v = xs.filter(x => Number.isFinite(x));
  if (!v.length) return null;
  const lo = Math.min(...v), hi = Math.max(...v), step = (hi - lo) / k;
  const counts = new Array(k).fill(0);
  for (const x of v) counts[step > 0 ? Math.min(Math.floor((x - lo) / step), k - 1) : 0]++;
  return { lo, hi, step, counts };
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
    d = 1 + aa * d;
    if (Math.abs(d) < BETACF_EPS) d = BETACF_EPS;
    c = 1 + aa / c;
    if (Math.abs(c) < BETACF_EPS) c = BETACF_EPS;
    d = 1 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < BETACF_EPS) d = BETACF_EPS;
    c = 1 + aa / c;
    if (Math.abs(c) < BETACF_EPS) c = BETACF_EPS;
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
  return p == null ? '' : p < SIG_P01 ? '**' : p < SIG_P05 ? '*' : p < SIG_P10 ? '.' : '';
}
function pairN(xs, ys) {
  let k = 0;
  for (let i = 0; i < Math.min(xs.length, ys.length); i++)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) k++;
  return k;
}
function corrEntry(r, n) {
  return { r, p: corrPvalue(r, n), sig: sigMark(r, n) };
}
function partialCorr(xs, ys, zs) {
  const rxy = pearson(xs, ys), rxz = pearson(xs, zs), ryz = pearson(ys, zs);
  if (rxy == null || rxz == null || ryz == null) return null;
  const den = Math.sqrt(Math.max(0, 1 - rxz * rxz) * Math.max(0, 1 - ryz * ryz));
  return den > 0 ? (rxy - rxz * ryz) / den : null;
}
function logLogSlope(ys, xs) {
  const p = [];
  for (let i = 0; i < Math.min(xs.length, ys.length); i++)
    if (Number.isFinite(xs[i]) && Number.isFinite(ys[i]) && xs[i] > 0 && ys[i] > 0)
      p.push([Math.log(xs[i]), Math.log(ys[i])]);
  const slopeOf = pts => {
    if (pts.length < 2) return null;
    const mx = pts.reduce((a, q) => a + q[0], 0) / pts.length;
    const my = pts.reduce((a, q) => a + q[1], 0) / pts.length;
    let sxy = 0, sxx = 0;
    for (const q of pts) { sxy += (q[0] - mx) * (q[1] - my); sxx += (q[0] - mx) ** 2; }
    return sxx > LOGLOG_EPS ? sxy / sxx : null;
  };
  if (p.length < 2) return { beta: null, betaRobust: null, leverage: null };
  const beta = slopeOf(p);
  if (beta == null) return { beta: null, betaRobust: null, leverage: null };
  if (p.length === 2) return { beta, betaRobust: null, leverage: null };
  const mx = p.reduce((a, q) => a + q[0], 0) / p.length;
  let wi = 0, wd = -1;
  for (let i = 0; i < p.length; i++) {
    const d = Math.abs(p[i][0] - mx);
    if (d > wd) { wd = d; wi = i; }
  }
  const betaRobust = slopeOf(p.filter((_, i) => i !== wi));
  if (betaRobust == null) return { beta, betaRobust: null, leverage: null };
  return { beta, betaRobust, leverage: Math.abs(beta - betaRobust) };
}
function skewKurt(xs) {
  const v = xs.filter(Number.isFinite);
  if (v.length < 3) return { skew: null, kurt: null };
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  let m2 = 0, m3 = 0, m4 = 0;
  for (const x of v) { const d = x - mean, d2 = d * d; m2 += d2; m3 += d2 * d; m4 += d2 * d2; }
  m2 /= v.length; m3 /= v.length; m4 /= v.length;
  if (m2 <= 0) return { skew: null, kurt: null };
  return { skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) - 3 };
}
function analyzeCorrelations(out, geo, m) {
  const skL = skewKurt(out.rows.map(r => r.reprojL)), skR = skewKurt(out.rows.map(r => r.reprojR)), skM = skewKurt(out.rows.map(viewMaxErr));
  const fxFyPct = {};
  for (const s of ['left', 'right']) {
    const K = m?.sides?.[s]?.K;
    fxFyPct[s] = K && K[0][0] > 0 ? Math.abs(K[0][0] - K[1][1]) / K[0][0] * 100 : null;
  }
  const base = {
    skew: { left: skL.skew, right: skR.skew, max: skM.skew },
    kurt: { left: skL.kurt, right: skR.kurt, max: skM.kurt },
    fxFyPct,
  };
  const usable = out.rows.filter(r => Number.isFinite(viewMaxErr(r)));
  if (!geo?.left) return {
    ...base,
    n: usable.length,
    corrDist: corrEntry(null, 0), corrAngle: corrEntry(null, 0), corrCoverInv: corrEntry(null, 0),
    corrDistPartial: corrEntry(null, 0), corrEdgePartial: corrEntry(null, 0),
    corrFrameIdx: corrEntry(pearson(usable.map(viewMaxErr), usable.map(r => r.view)), usable.length),
    coverage: null,
  };
  const sel = out.rows.filter(r => Number.isFinite(r.reprojL) && geo.left.perView[r.view] != null);
  const err = sel.map(r => r.reprojL);
  const dist = sel.map(r => r.distL), angle = sel.map(r => r.angleL), cover = sel.map(r => r.coverL);
  const n = err.length;
  return {
    ...base,
    n,
    corrDist: corrEntry(pearson(err, dist), n),
    corrAngle: corrEntry(pearson(err, angle), n),
    corrCoverInv: corrEntry(pearson(err, cover.map(x => -x)), n),
    corrDistPartial: corrEntry(partialCorr(err, dist, angle), n),
    corrEdgePartial: corrEntry(partialCorr(err, angle, dist), n),
    corrFrameIdx: corrEntry(pearson(err, sel.map(r => r.view)), n),
    coverage: { dist: bins(dist, COVERAGE_BINS), angle: bins(angle, COVERAGE_BINS), cover: bins(cover, COVERAGE_BINS) },
  };
}
function stereoConsistency(m) {
  const lv = m.sides?.left?.perView ?? [], rv = m.sides?.right?.perView ?? [];
  const epi = m.extrinsic?.epiPerView ?? [];
  const nLR = Math.min(lv.length, rv.length);
  const diffs = [], maxLR = [];
  for (let i = 0; i < nLR; i++) {
    if (!Number.isFinite(lv[i]) || !Number.isFinite(rv[i])) { maxLR.push(NaN); continue; }
    diffs.push(lv[i] - rv[i]);
    maxLR.push(Math.max(lv[i], rv[i]));
  }
  const mean = diffs.length ? diffs.reduce((a, b) => a + b, 0) / diffs.length : null;
  const std = diffs.length ? Math.sqrt(diffs.reduce((a, b) => a + (b - mean) ** 2, 0) / diffs.length) : null;
  const betaOf = side => {
    const sd = m.sides?.[side];
    if (!sd) return { beta: null, betaRobust: null, leverage: null };
    const pv = sd.perView ?? [], tv = sd.tvecs ?? [];
    const k = Math.min(pv.length, tv.length), xs = [];
    for (let i = 0; i < k; i++) { const t = vec3(tv[i]); xs.push(t ? vecNorm(t) : NaN); }
    return logLogSlope(pv, xs);
  };
  let leftWorse = 0, rightWorse = 0, cmp = 0;
  for (let i = 0; i < nLR; i++) {
    if (!Number.isFinite(lv[i]) || !Number.isFinite(rv[i]) || lv[i] <= 0 || rv[i] <= 0) continue;
    cmp++;
    const ratio = lv[i] / rv[i];
    if (ratio > IMBALANCE_HI) leftWorse++;
    else if (ratio < IMBALANCE_LO) rightWorse++;
  }
  const imb = leftWorse + rightWorse;
  return {
    lrDiff: { mean, std },
    corrLR: corrEntry(pearson(lv, rv), pairN(lv, rv)),
    corrReprojEpi: corrEntry(pearson(maxLR, epi), pairN(maxLR, epi)),
    betaL: betaOf('left'), betaR: betaOf('right'),
    imbalancedFrames: { count: imb, pct: cmp ? imb / cmp * 100 : null, leftWorse, rightWorse },
  };
}
function coverageAudit(m, geo) {
  const na = reason => ({
    frames: { value: null, ok: null, reason },
    orientationDiversity: { value: null, ok: null, reason },
    regionCoverage: { value: null, ok: null, reason },
    ok: null, reason,
  });
  if (!geo) return na('no board config');
  if (!geo.left) return na('no left side geometry');
  const W = m.imageSize?.[0], H = m.imageSize?.[1];
  const fv = m.frames?.valid;
  const frames = { value: fv, ok: fv != null ? fv >= MIN_VALID_FRAMES : null };
  const rvs = m.sides?.left?.rvecs ?? [];
  const pvGeo = geo.left.perView ?? [];
  const normals = [];
  for (let i = 0; i < Math.min(rvs.length, pvGeo.length); i++) {
    if (pvGeo[i] == null) continue;
    const r = vec3(rvs[i]);
    if (!r) continue;
    const R = rodrigues(r);
    normals.push([R[0][2], R[1][2], R[2][2]]);
  }
  let orient = { value: null, ok: null, reason: 'no valid pose' };
  if (normals.length) {
    let deviating = 0;
    for (let i = 1; i < normals.length; i++) {
      const dot = normals[0][0] * normals[i][0] + normals[0][1] * normals[i][1] + normals[0][2] * normals[i][2];
      if (Math.acos(Math.max(-1, Math.min(1, dot))) * 180 / Math.PI > ORIENT_ANGLE_DEG) deviating++;
    }
    orient = { value: deviating, ok: deviating >= 1 };
  }
  const grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const v of pvGeo) {
    if (!v || !Number.isFinite(v.uc) || !Number.isFinite(v.vc)) continue;
    const c = Math.min(2, Math.max(0, Math.floor(v.uc / (W / 3))));
    const r = Math.min(2, Math.max(0, Math.floor(v.vc / (H / 3))));
    grid[r][c]++;
  }
  const cellEmpty = (r, c) => grid[r][c] === 0 ? 1 : 0;
  const cornerEmpty = cellEmpty(0, 0) + cellEmpty(0, 2) + cellEmpty(2, 0) + cellEmpty(2, 2);
  const edgeEmpty = cellEmpty(0, 1) + cellEmpty(1, 0) + cellEmpty(1, 2) + cellEmpty(2, 1);
  return {
    frames,
    orientationDiversity: orient,
    regionCoverage: { value: { grid, cornerEmpty, edgeEmpty }, ok: cornerEmpty <= CORNER_EMPTY_MAX },
    ok: frames.ok === true && orient.ok === true && cornerEmpty <= CORNER_EMPTY_MAX,
  };
}
const sigStars = s => typeof s === 'string' ? (s.match(/\*/g) ?? []).length : 0;
function matchFingerprints(m) {
  const ste = m._stats?.stereo ?? null;
  const cov = m._stats?.coverage ?? null;
  const corr = m._corr ?? null;
  const rules = [];
  const add = (id, name, verdict, evidence) => rules.push({ id, name, verdict, evidence });
  const bL = ste?.betaL?.beta ?? null, bR = ste?.betaR?.beta ?? null;
  if (bL == null || bR == null)
    add('R1', '双目对焦/距离响应不一致', 'insufficient', {
      betaL: bL, betaR: bR,
      note: `β${bL == null ? 'L' : ''}${bR == null ? 'R' : ''} 缺失（无逐帧误差或平移向量）`,
    });
  else
    add('R1', '双目对焦/距离响应不一致',
      Math.abs(bL - bR) > FP_BETA_GAP && bL * bR < 0 ? 'hit' : 'clear', { betaL: bL, betaR: bR });
  const covSub = {
    frames: cov?.frames?.ok ?? null,
    orientation: cov?.orientationDiversity?.ok ?? null,
    region: cov?.regionCoverage?.ok ?? null,
  };
  if (cov?.ok === false)
    add('R2', '姿态覆盖不足', 'hit', covSub);
  else if (cov?.ok === true)
    add('R2', '姿态覆盖不足', 'clear', covSub);
  else
    add('R2', '姿态覆盖不足', 'insufficient', { ...covSub, note: `覆盖审计不可测（${cov?.reason ?? 'geo 缺失'}）` });
  const cd = corr?.corrDist ?? null;
  if (cd?.r == null || bL == null)
    add('R3', '远距小板亚像素不足', 'insufficient', {
      corrDist: cd, betaL: bL,
      note: cd?.r == null ? 'corrDist 缺失（geo 缺失）' : 'βL 缺失',
    });
  else
    add('R3', '远距小板亚像素不足',
      cd.r > FP_CORR_STRONG && sigStars(cd.sig) >= FP_CORR_SIGSTAR && bL >= FP_BETA_PHYS_LO && bL <= FP_BETA_PHYS_HI ? 'hit' : 'clear',
      { corrDist: cd, betaL: bL });
  const epiMean = m.extrinsic?.epiMean ?? null, repMean = m.reproj?.mean ?? null;
  const cre = ste?.corrReprojEpi ?? null;
  const epiRatio = epiMean != null && repMean != null && repMean > 0 ? epiMean / repMean : null;
  if (epiRatio == null || cre?.r == null)
    add('R4', '外参/装配嫌疑', 'insufficient', {
      epiReprojRatio: epiRatio, corrReprojEpi: cre,
      note: `${epiRatio == null ? 'epiMean/reprojMean' : 'corrReprojEpi'} 缺失`,
    });
  else
    add('R4', '外参/装配嫌疑',
      epiRatio > FP_EPI_RATIO && Math.abs(cre.r) < FP_EPI_DECOUPLE ? 'hit' : 'clear',
      { epiReprojRatio: epiRatio, corrReprojEpi: cre });
  const imb = ste?.imbalancedFrames ?? null;
  if (imb?.pct == null)
    add('R5', '单相机劣化', 'insufficient', { pct: null, leftWorse: imb?.leftWorse ?? null, rightWorse: imb?.rightWorse ?? null, note: '失衡帧统计缺失' });
  else
    add('R5', '单相机劣化',
      imb.pct > FP_IMBAL_PCT && ((imb.leftWorse ?? 0) >= FP_DOMINANCE * (imb.rightWorse ?? 0) || (imb.rightWorse ?? 0) >= FP_DOMINANCE * (imb.leftWorse ?? 0)) ? 'hit' : 'clear',
      { pct: imb.pct, leftWorse: imb.leftWorse ?? 0, rightWorse: imb.rightWorse ?? 0 });
  const clr = ste?.corrLR ?? null;
  if (clr?.r == null) add('R6', '全局场景问题', 'insufficient', { corrLR: clr, note: 'corrLR 缺失' });
  else add('R6', '全局场景问题', clr.r > FP_LR_STRONG && sigStars(clr.sig) >= FP_CORR_SIGSTAR ? 'hit' : 'clear', { corrLR: clr });
  const cfi = corr?.corrFrameIdx ?? null;
  if (cfi?.r == null) add('R7', '采集期热漂', 'insufficient', { corrFrameIdx: cfi, note: 'corrFrameIdx 缺失' });
  else add('R7', '采集期热漂', Math.abs(cfi.r) > FP_DRIFT && sigStars(cfi.sig) >= FP_CORR_SIGSTAR ? 'hit' : 'clear', { corrFrameIdx: cfi });
  const vf = m.frames?.valid ?? null;
  const oOk = cov?.orientationDiversity?.ok ?? null;
  if (vf != null && vf < FP_MIN_FRAMES)
    add('R8', '数据量/姿态病态', 'hit', { frames: vf, orientation: oOk });
  else if (vf == null && oOk == null)
    add('R8', '数据量/姿态病态', 'insufficient', { frames: null, orientation: null, note: 'valid 帧数与姿态多样性均缺失' });
  else if (vf == null || oOk == null)
    add('R8', '数据量/姿态病态', 'insufficient', { frames: vf, orientation: oOk, note: vf == null ? 'valid 帧数不可测' : '姿态多样性不可测（geo 缺失）' });
  else
    add('R8', '数据量/姿态病态', oOk === false ? 'hit' : 'clear', { frames: vf, orientation: oOk });
  return { hits: rules.filter(r => r.verdict === 'hit').map(r => r.id), rules };
}
function fingerprintRecurrence(ms) {
  const out = [];
  for (const r of ms[0]._fingerprints?.rules ?? []) {
    const runs = ms.filter(m => m._fingerprints?.hits?.includes(r.id)).map(m => m.label);
    if (runs.length >= 2) out.push({ id: r.id, runs, note: runs.length >= 4 ? '跨批稳定重现（结论级）' : '初步重现（线索级，≥4 批=结论级）' });
  }
  return out;
}
function projectFieldDiff(m1, m2, side) {
  const s1 = m1.sides?.[side], s2 = m2.sides?.[side];
  const W = m1.imageSize?.[0], H = m1.imageSize?.[1];
  if (!s1?.K || !s2?.K || !s1.D || !s2.D || !W || !H) return null;
  const pad = d => [d[0] ?? 0, d[1] ?? 0, d[4] ?? 0, d[2] ?? 0, d[3] ?? 0];
  const d1 = pad(s1.D), d2 = pad(s2.D);
  const fx1 = s1.K[0][0], fy1 = s1.K[1][1], cx1 = s1.K[0][2], cy1 = s1.K[1][2];
  const fx2 = s2.K[0][0], fy2 = s2.K[1][1], cx2 = s2.K[0][2], cy2 = s2.K[1][2];
  const distort = (x, y, D) => {
    const r2 = x * x + y * y, rad = 1 + D[0] * r2 + D[1] * r2 ** 2 + D[2] * r2 ** 3;
    return [x * rad + 2 * D[3] * x * y + D[4] * (r2 + 2 * x * x), y * rad + D[3] * (r2 + 2 * y * y) + 2 * D[4] * x * y];
  };
  const undistort = (x, y, D) => {
    if (D.every(v => v === 0)) return [x, y];
    let ux = x, uy = y;
    for (let it = 0; it < UNDISTORT_MAX_IT; it++) {
      const r2 = ux * ux + uy * uy;
      const rad = 1 + D[0] * r2 + D[1] * r2 ** 2 + D[2] * r2 ** 3;
      const tx = 2 * D[3] * ux * uy + D[4] * (r2 + 2 * ux * ux);
      const ty = D[3] * (r2 + 2 * uy * uy) + 2 * D[4] * ux * uy;
      const nx = (x - tx) / rad, ny = (y - ty) / rad;
      const step = Math.max(Math.abs(nx - ux), Math.abs(ny - uy));
      ux = nx; uy = ny;
      if (step < UNDISTORT_EPS) break;
    }
    return [ux, uy];
  };
  let maxPx = 0, sumPx = 0, kOnlyMaxPx = 0, cnt = 0;
  for (let iy = 0; iy < GRID_NY; iy++) {
    for (let ix = 0; ix < GRID_NX; ix++) {
      const gx = -GRID_RANGE + ix * (2 * GRID_RANGE / (GRID_NX - 1));
      const gy = -GRID_RANGE + iy * (2 * GRID_RANGE / (GRID_NY - 1));
      const u = W / 2 + gx * (W / 2 - GRID_MARGIN_PX), v = H / 2 + gy * (H / 2 - GRID_MARGIN_PX);
      const xn = (u - cx1) / fx1, yn = (v - cy1) / fy1;
      const [xu, yu] = undistort(xn, yn, d1);
      const [xf, yf] = distort(xu, yu, d2);
      const df = Math.hypot(fx2 * xf + cx2 - u, fy2 * yf + cy2 - v);
      const [xk, yk] = distort(xu, yu, d1);
      kOnlyMaxPx = Math.max(kOnlyMaxPx, Math.hypot(fx2 * xk + cx2 - u, fy2 * yk + cy2 - v));
      maxPx = Math.max(maxPx, df);
      sumPx += df;
      cnt++;
    }
  }
  return {
    maxPx, meanPx: cnt ? sumPx / cnt : null, kOnlyMaxPx,
    paramDelta: {
      dfxPct: fx1 ? (fx2 - fx1) / fx1 * 100 : null, dcx: cx2 - cx1, dcy: cy2 - cy1,
      dk1: d2[0] - d1[0], dk2: d2[1] - d1[1], dk3: d2[2] - d1[2],
      dp1: d2[3] - d1[3], dp2: d2[4] - d1[4],
    },
  };
}

function extractMetrics(run) {
  const j = run.raw;
  if (!j || typeof j !== 'object') return { label: run.label, path: run.path, fatal: 'top-level not an object' };
  const m = { label: run.label, path: run.path };
  const warn = [];
  if (j.schema !== 'factory_calib.camera_calib.v1') warn.push(`schema=${j.schema}`);
  m.schema = j.schema;
  m.imageSize = Array.isArray(j.imageSize) ? j.imageSize : null;
  m.referenceTemp = num(j.referenceTemp);
  m.cte = num(j.cte);
  m.frames = { input: num(j.intrinsic?.total_frames_input), valid: num(j.intrinsic?.valid_frames_count) };
  if (m.frames.input && m.frames.valid != null) m.frames.skippedPct = (1 - m.frames.valid / m.frames.input) * 100;

  const sides = {};
  for (const s of ['left', 'right']) {
    const o = j.intrinsic?.[s];
    sides[s] = o ? {
      rms: num(o.rms_error),
      K: mat3(o.camera_matrix),
      D: arr(Array.isArray(o.dist_coeffs?.[0]) ? o.dist_coeffs[0] : o.dist_coeffs),
      perView: arr(o.per_view_errors),
      rvecs: Array.isArray(o.rvecs) ? o.rvecs : [],
      tvecs: Array.isArray(o.tvecs) ? o.tvecs : [],
      viewStats: stats(arr(o.per_view_errors)),
    } : null;
  }
  m.sides = sides;
  m.reproj = { mean: num(j.intrinsic?.reproj_error_mean), std: num(j.intrinsic?.reproj_error_std) };

  const e = j.extrinsic;
  m.extrinsic = e ? {
    stereoRms: num(e.stereoReprojError),
    epiMean: num(e.epipolarErrorMean), epiStd: num(e.epipolarErrorStd),
    epiPerView: arr(e.perViewEpipolarErrors),
    epiStats: stats(arr(e.perViewEpipolarErrors)),
    T: matCol(e.T), R: mat3(e.R), qualityFlag: e.qualityFlag, success: e.success,
  } : null;
  m.baseline = m.extrinsic?.T ? vecNorm(m.extrinsic.T) : null;

  const r = j.rectify;
  const roi = o => o && num(o.w ?? o.width) != null && num(o.h ?? o.height) != null
    ? { x: num(o.x), y: num(o.y), width: num(o.w ?? o.width), height: num(o.h ?? o.height) } : null;
  const W = m.imageSize?.[0], H = m.imageSize?.[1];
  m.roi = r ? { L: roi(r.validRoiLeft), R: roi(r.validRoiRight) } : null;
  if (m.roi && W && H) {
    const cov = o => o ? (o.width * o.height) / (W * H) * 100 : null;
    m.roi.coverL = cov(m.roi.L); m.roi.coverR = cov(m.roi.R);
  }

  const fx = s => sides[s]?.K ? (sides[s].K[0][0] + sides[s].K[1][1]) / 2 : null;
  const cx = s => sides[s]?.K ? sides[s].K[0][2] : null;
  const cy = s => sides[s]?.K ? sides[s].K[1][2] : null;
  const fl = fx('left'), fr = fx('right');
  m.consistency = {
    focalL: fl, focalR: fr,
    focalDiffPct: fl && fr ? Math.abs(fl - fr) / ((fl + fr) / 2) * 100 : null,
    cxL: cx('left'), cyL: cy('left'), cxR: cx('right'), cyR: cy('right'),
  };
  if (W && H) {
    if (m.consistency.cxL != null && m.consistency.cyL != null)
      m.consistency.principalOffL = Math.hypot(m.consistency.cxL - W / 2, m.consistency.cyL - H / 2);
    if (m.consistency.cxR != null && m.consistency.cyR != null)
      m.consistency.principalOffR = Math.hypot(m.consistency.cxR - W / 2, m.consistency.cyR - H / 2);
  }

  const tables = {};
  for (const name of ['intrinsicTempTableL', 'intrinsicTempTableR', 'extrinsicTempTable', 'stereoRectifyTempTable']) {
    const t = j[name];
    if (!t) { tables[name] = { present: false }; continue; }
    const wrap = Array.isArray(t) ? t[0] : t;
    const size = num(wrap?.tableSize) ?? (Array.isArray(wrap?.table) ? wrap.table.length : null);
    tables[name] = { present: true, tiers: size, success: wrap?.success ?? null,
      refTemp: num(wrap?.referenceTemp) ?? null };
  }
  m.tempTables = tables;
  m.warnings = warn;
  return m;
}

function resolveThresholds(args, run) {
  let reproj = args.reprojTh, src = 'default';
  if (reproj != null) return { reproj, epipolar: args.epipolarTh ?? 0.05, reprojSrc: 'cli', epiSrc: args.epipolarTh != null ? 'cli' : 'default' };
  try {
    const cfg = JSON.parse(readFileSync(join(dirname(run.path), 'config.json'), 'utf8'));
    const v = num(cfg.reproj_error_threshold);
    if (v != null) { reproj = v; src = 'config'; }
  } catch {}
  return { reproj: reproj ?? 0.012, epipolar: args.epipolarTh ?? 0.05, reprojSrc: src, epiSrc: args.epipolarTh != null ? 'cli' : 'default' };
}

function gateMetrics(m, th) {
  const g = [];
  const add = (name, value, limit) => {
    if (!Number.isFinite(value)) return g.push({ name, value: null, limit, verdict: 'N/A' });
    g.push({ name, value, limit, verdict: value > limit ? 'FAIL' : 'PASS' });
  };
  add('reproj_mean', m.reproj?.mean, th.reproj);
  add('epipolar_mean', m.extrinsic?.epiMean, th.epipolar);
  g.push(Number.isFinite(m.frames.skippedPct)
    ? { name: 'skipped_pct', value: m.frames.skippedPct, limit: 10, verdict: m.frames.skippedPct > 10 ? 'FAIL' : m.frames.skippedPct > 3 ? 'WARN' : 'PASS' }
    : { name: 'skipped_pct', value: null, limit: 10, verdict: 'N/A' });
  for (const name of Object.keys(m.tempTables)) {
    const t = m.tempTables[name];
    if (!t.present) g.push({ name: `table:${name}`, verdict: 'FAIL' });
    else if (t.tiers !== 61) g.push({ name: `table:${name}`, value: t.tiers, limit: 61, verdict: 'WARN' });
  }
  const order = { FAIL: 0, WARN: 1, 'N/A': 2, PASS: 3 };
  m.gates = g.sort((a, b) => order[a.verdict] - order[b.verdict]);
  m.verdict = m.gates.some(x => x.verdict === 'FAIL') ? 'FAIL'
    : m.gates.some(x => x.verdict === 'WARN') ? 'WARN' : 'PASS';
  return m;
}

const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function svgDoc(w, h, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`
    + `<rect x="0" y="0" width="${w}" height="${h}" fill="#fff"/>`
    + `<g font-family="Consolas, monospace" font-size="11">${body}</g></svg>`;
}
function chartFrame(w, h, pad, title, xlab, ylab) {
  const x0 = pad.l, y0 = pad.t, x1 = w - pad.r, y1 = h - pad.b;
  const cx = ((x0 + x1) / 2).toFixed(1), cy = ((y0 + y1) / 2).toFixed(1);
  const body = `<text x="${cx}" y="${(y0 - 14).toFixed(1)}" text-anchor="middle" fill="#222" font-size="13">${esc(title)}</text>`
    + `<line x1="${x0}" y1="${y1}" x2="${x1}" y2="${y1}" stroke="#555"/>`
    + `<line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y1}" stroke="#555"/>`
    + `<text x="${cx}" y="${(h - 8).toFixed(1)}" text-anchor="middle" fill="#444">${esc(xlab)}</text>`
    + `<text x="12" y="${cy}" text-anchor="middle" fill="#444" transform="rotate(-90 12 ${cy})">${esc(ylab)}</text>`;
  return { body, x0, y0, x1, y1 };
}
function scale(v, d0, d1, r0, r1) {
  return d1 === d0 ? (r0 + r1) / 2 : r0 + (v - d0) / (d1 - d0) * (r1 - r0);
}
function fmtTick(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
}
function gridY(x0, x1, y0, y1, d0, d1, fmtFn) {
  let s = '';
  for (let k = 1; k <= 4; k++) {
    const yv = d0 + (d1 - d0) * k / 4;
    const yy = scale(yv, d0, d1, y1, y0).toFixed(1);
    s += `<line x1="${x0}" y1="${yy}" x2="${x1}" y2="${yy}" stroke="#ddd"/>`
      + `<text x="${x0 - 6}" y="${(+yy + 4).toFixed(1)}" text-anchor="end" fill="#555">${fmtFn(yv)}</text>`;
  }
  return s;
}
function svgBars(title, xlab, ylab, series, opts = {}) {
  const w = 860, h = 380, pad = { l: 72, r: 18, t: 34, b: 46 };
  const { body: fr, x0, y0, x1, y1 } = chartFrame(w, h, pad, title, xlab, ylab);
  const n = Math.max(0, ...series.map(s => s.values.length));
  const lines = (opts.lines ?? []).filter(l => l.value != null && Number.isFinite(l.value));
  const all = series.flatMap(s => s.values).filter(v => v != null && Number.isFinite(v));
  let dmax = Math.max(0, ...all, ...lines.map(l => l.value));
  if (!(dmax > 0)) dmax = 1;
  let body = fr + gridY(x0, x1, y0, y1, 0, dmax, fmtTick)
    + `<text x="${x0 - 6}" y="${(y1 + 4).toFixed(1)}" text-anchor="end" fill="#555">0</text>`;
  for (const l of lines) {
    const yy = scale(l.value, 0, dmax, y1, y0).toFixed(1);
    body += `<line x1="${x0}" y1="${yy}" x2="${x1}" y2="${yy}" stroke="${l.color}" stroke-dasharray="6 4"/>`
      + `<text x="${x1}" y="${(+yy - 5).toFixed(1)}" text-anchor="end" fill="${l.color}">${esc(`${l.label} ${fmtTick(l.value)}`)}</text>`;
  }
  const groupW = n > 0 ? (x1 - x0) / n : 0, slot = groupW / series.length;
  for (const [si, s] of series.entries()) {
    for (let i = 0; i < s.values.length; i++) {
      const v = s.values[i];
      if (v == null || !Number.isFinite(v)) continue;
      const bx = x0 + i * groupW + si * slot + slot * 0.12;
      const by = scale(v, 0, dmax, y1, y0);
      body += `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${Math.max(1, slot * 0.76).toFixed(1)}" height="${Math.max(0.5, y1 - by).toFixed(1)}" fill="${s.color}"/>`;
    }
  }
  const stepI = Math.max(1, Math.ceil(n / 22));
  for (let i = 0; i < n; i += stepI)
    body += `<text x="${(x0 + (i + 0.5) * groupW).toFixed(1)}" y="${(y1 + 14).toFixed(1)}" text-anchor="middle" fill="#555">${i + 1}</text>`;
  let lx = x1;
  for (const s of [...series].reverse()) {
    lx -= s.name.length * 6.6 + 26;
    body += `<rect x="${lx.toFixed(1)}" y="${(pad.t - 20).toFixed(1)}" width="10" height="10" fill="${s.color}"/>`
      + `<text x="${(lx + 14).toFixed(1)}" y="${(pad.t - 11).toFixed(1)}" fill="#333">${esc(s.name)}</text>`;
  }
  return svgDoc(w, h, body);
}
function svgHist(title, xlab, counts, lo, step) {
  const w = 560, h = 300, pad = { l: 56, r: 14, t: 30, b: 42 };
  const { body: fr, x0, x1, y1, y0 } = chartFrame(w, h, pad, title, xlab, '帧数');
  const k = counts.length, maxC = Math.max(1, ...counts);
  let body = fr + gridY(x0, x1, y0, y1, 0, maxC, v => String(Math.round(v)));
  const bw = (x1 - x0) / k;
  const xd = step >= 10 ? 0 : step >= 1 ? 1 : step >= 0.1 ? 2 : 3;
  counts.forEach((c, i) => {
    const bh = scale(c, 0, maxC, 0, y1 - y0);
    body += `<rect x="${(x0 + i * bw + bw * 0.08).toFixed(1)}" y="${(y1 - bh).toFixed(1)}" width="${(bw * 0.84).toFixed(1)}" height="${Math.max(0.5, bh).toFixed(1)}" fill="#468"/>`;
    if (i % 2 === 0)
      body += `<text x="${(x0 + (i + 0.5) * bw).toFixed(1)}" y="${(y1 + 14).toFixed(1)}" text-anchor="middle" fill="#555">${(lo + (i + 0.5) * step).toFixed(xd)}</text>`;
  });
  return svgDoc(w, h, body);
}
function svgScatter(title, xlab, ylab, pts) {
  const w = 560, h = 300, pad = { l: 56, r: 14, t: 30, b: 42 };
  const { body: fr, x0, x1, y0, y1 } = chartFrame(w, h, pad, title, xlab, ylab);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const mx = (xmax - xmin) * 0.05 || 0.5, my = (ymax - ymin) * 0.05 || 0.005;
  const dx0 = xmin - mx, dx1 = xmax + mx, dy0 = ymin - my, dy1 = ymax + my;
  let body = fr + gridY(x0, x1, y0, y1, dy0, dy1, fmtTick);
  for (const [px, py] of pts)
    body += `<circle cx="${scale(px, dx0, dx1, x0, x1).toFixed(1)}" cy="${scale(py, dy0, dy1, y1, y0).toFixed(1)}" r="2.6" fill="#468" fill-opacity="0.75"/>`;
  body += `<text x="${x0}" y="${(y1 + 14).toFixed(1)}" text-anchor="start" fill="#555">${fmtTick(xmin)}</text>`
    + `<text x="${x1}" y="${(y1 + 14).toFixed(1)}" text-anchor="end" fill="#555">${fmtTick(xmax)}</text>`;
  return svgDoc(w, h, body);
}
function svgCompare(title, rows) {
  const w = 560, palette = ['#4682b4', '#b47846', '#48a048', '#a04884', '#6666aa', '#aa6666'];
  const items0 = rows[0]?.items ?? [], rowH = 15;
  const h = 46 + rows.length * (items0.length * rowH + 26) + 6;
  let body = `<text x="${(w / 2).toFixed(1)}" y="18" text-anchor="middle" fill="#222" font-size="13">${esc(title)}</text>`;
  let lx = 12;
  items0.forEach((it, j) => {
    body += `<rect x="${lx}" y="26" width="10" height="10" fill="${palette[j % palette.length]}"/>`
      + `<text x="${lx + 14}" y="35" fill="#333">${esc(it.label)}</text>`;
    lx += 14 + it.label.length * 6.6 + 16;
  });
  let y = 46;
  const x0m = 132, x1m = 452;
  for (const r of rows) {
    body += `<text x="12" y="${(y + 4 + items0.length * rowH / 2).toFixed(1)}" fill="#222">${esc(r.metric)}</text>`;
    const maxV = Math.max(0, ...r.items.map(it => it.value != null && Number.isFinite(it.value) ? it.value : -Infinity));
    r.items.forEach((it, j) => {
      const by = y + 4 + j * rowH;
      if (it.value == null || !Number.isFinite(it.value) || maxV <= 0) {
        body += `<text x="${x0m}" y="${(by + 9).toFixed(1)}" fill="#777">${it.value == null || !Number.isFinite(it.value) ? 'N/A' : it.value.toFixed(3)}</text>`;
      } else {
        const bw2 = Math.max(1, scale(it.value, 0, maxV, 0, x1m - x0m));
        body += `<rect x="${x0m}" y="${by}" width="${bw2.toFixed(1)}" height="9" fill="${palette[j % palette.length]}"/>`
          + `<text x="${(x0m + bw2 + 5).toFixed(1)}" y="${(by + 9).toFixed(1)}" fill="#333">${it.value.toFixed(3)}</text>`;
      }
    });
    y += items0.length * rowH + 26;
  }
  return svgDoc(w, h, body);
}
function buildCharts(m, out, corr, label) {
  const charts = {};
  charts.perframe = svgBars(`${label} 逐帧误差`, 'view#', '误差 (px)', [
    { name: 'reproj L', color: '#468', values: out.rows.map(r => r.reprojL) },
    { name: 'reproj R', color: '#864', values: out.rows.map(r => r.reprojR) },
    { name: 'epipolar', color: '#484', values: out.rows.map(r => r.epi) },
  ], { lines: [
    { value: m._th?.reproj, color: '#c33', label: 'reproj 阈值' },
    { value: m._th?.epipolar, color: '#36c', label: 'epipolar 阈值' },
  ] });
  const hb = bins(m.sides?.left?.perView ?? [], 20);
  charts.hist = hb ? svgHist(`${label} reproj L 分布`, '误差 (px)', hb.counts, hb.lo, hb.step) : null;
  if (corr?.coverage) {
    const ptD = [], ptA = [];
    for (const r of out.rows) {
      const e = viewMaxErr(r);
      if (e == null || !Number.isFinite(e)) continue;
      if (r.distL != null) ptD.push([r.distL, e]);
      if (r.angleL != null) ptA.push([r.angleL, e]);
    }
    charts.scatterDist = ptD.length ? svgScatter(`${label} 误差~距离`, '距离 (mm)', 'max(reproj) (px)', ptD) : null;
    charts.scatterAngle = ptA.length ? svgScatter(`${label} 误差~斜角`, '斜角 (°)', 'max(reproj) (px)', ptA) : null;
    const cd = corr.coverage.dist, ca = corr.coverage.angle;
    charts.coverDist = cd ? svgHist(`${label} 距离分布`, '距离 (mm)', cd.counts, cd.lo, cd.step) : null;
    charts.coverAngle = ca ? svgHist(`${label} 斜角分布`, '斜角 (°)', ca.counts, ca.lo, ca.step) : null;
  }
  return charts;
}

function fmt(v, d = 4, unit = '') {
  return v == null || !Number.isFinite(v) ? 'N/A' : v.toFixed(d) + unit;
}
function now() {
  const d = new Date(), z = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${z(d.getMonth() + 1)}-${z(d.getDate())} ${z(d.getHours())}:${z(d.getMinutes())}:${z(d.getSeconds())}`;
}
const mdEsc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const viewList = a => a?.length ? a.map(i => i + 1).join(', ') : 'N/A';
const FP_MARK = { hit: '⚠命中', clear: '—', insufficient: '? 证据不足' };
const fpSubMark = v => v === true ? '✓' : v === false ? '✗' : '?';
function fpSummary(r) {
  const e = r.evidence ?? {};
  if (r.verdict === 'insufficient') return e.note ?? '证据不足';
  const f3 = v => v == null ? 'N/A' : (v >= 0 ? '+' : '') + v.toFixed(3);
  switch (r.id) {
    case 'R1': return `βL=${f3(e.betaL)} βR=${f3(e.betaR)}`;
    case 'R2': return `帧数${fpSubMark(e.frames)} 方向${fpSubMark(e.orientation)} 宫格${fpSubMark(e.region)}`;
    case 'R3': return `corrDist=${f3(e.corrDist?.r)}${e.corrDist?.sig ?? ''} βL=${f3(e.betaL)}`;
    case 'R4': return `epi/reproj=${e.epiReprojRatio.toFixed(2)} corrReprojEpi=${f3(e.corrReprojEpi?.r)}${e.corrReprojEpi?.sig ?? ''}`;
    case 'R5': return `失衡${e.pct.toFixed(1)}% 左${e.leftWorse}/右${e.rightWorse}`;
    case 'R6': return `corrLR=${f3(e.corrLR?.r)}${e.corrLR?.sig ?? ''}`;
    case 'R7': return `corrFrameIdx=${f3(e.corrFrameIdx?.r)}${e.corrFrameIdx?.sig ?? ''}`;
    case 'R8': return `valid=${e.frames ?? 'N/A'} 方向${fpSubMark(e.orientation)}`;
    default: return '';
  }
}
function renderGlossary() {
  const L = ['## 指标说明', ''];
  for (const g of GLOSSARY) {
    L.push(`### ${mdEsc(g.group)}`, '');
    L.push('| 指标 | 通俗含义 |', '|---|---|');
    for (const [term, meaning] of g.rows) L.push(`| ${mdEsc(term)} | ${mdEsc(meaning)} |`);
    L.push('');
  }
  return L.join('\n');
}
function renderRunSection(m, out, corr, th) {
  const L = [];
  L.push(`## ${mdEsc(m.label)}`, '');
  L.push(`- 文件：\`${mdEsc(m.path)}\``);
  L.push(`- schema：${mdEsc(m.schema ?? 'N/A')}${m.warnings?.length ? `（⚠ ${mdEsc(m.warnings.join('；'))}）` : ''}`);
  L.push(`- 参考温度：${fmt(m.referenceTemp, 1, ' °C')}`);
  L.push(`- 帧：input ${fmt(m.frames?.input, 0)} / valid ${fmt(m.frames?.valid, 0)} / 跳过 ${fmt(m.frames?.skippedPct, 1, '%')}`);
  L.push(`- 基线 |T|：${fmt(m.baseline, 2, ' mm')}`);
  L.push(`- 阈值：reproj ≤ ${th.reproj}（${th.reprojSrc}），epipolar ≤ ${th.epipolar}（${th.epiSrc}）`);
  L.push(`- 综合判定：**${m.verdict}**`, '');
  L.push('### 质量指标', '', '| 指标 | 值 | 阈值 | 判定 |', '|---|---|---|---|');
  for (const g of m.gates ?? []) {
    const val = g.value == null ? 'N/A' : g.name === 'skipped_pct' ? fmt(g.value, 1, '%')
      : String(g.name).startsWith('table:') ? String(g.value) : fmt(g.value);
    const lim = g.limit == null ? 'N/A' : g.name === 'skipped_pct' ? `≤ ${g.limit}%` : `≤ ${g.limit}`;
    L.push(`| ${g.name} | ${val} | ${lim} | ${g.verdict} |`);
  }
  L.push('');
  L.push('### 分布（per-view）', '', '| 序列 | n | min | P50 | P90 | max | mean | std |', '|---|---|---|---|---|---|---|---|');
  for (const [nm, s] of [['reproj L', m.sides?.left?.viewStats], ['reproj R', m.sides?.right?.viewStats], ['epipolar', m.extrinsic?.epiStats]])
    L.push(`| ${nm} | ${s ? s.n : 'N/A'} | ${fmt(s?.min)} | ${fmt(s?.p50)} | ${fmt(s?.p90)} | ${fmt(s?.max)} | ${fmt(s?.mean)} | ${fmt(s?.std)} |`);
  L.push('');
  const c = m.consistency ?? {};
  L.push('### 一致性', '');
  L.push(`- 焦距：L ${fmt(c.focalL, 2, ' px')} / R ${fmt(c.focalR, 2, ' px')}（Δ ${fmt(c.focalDiffPct, 2, '%')}）`);
  L.push(`- 主点偏离中心：L ${fmt(c.principalOffL, 2, ' px')} / R ${fmt(c.principalOffR, 2, ' px')}`);
  L.push(`- validRoi 覆盖：L ${fmt(m.roi?.coverL, 1, '%')} / R ${fmt(m.roi?.coverR, 1, '%')}`, '');
  L.push('### 离群帧', '');
  L.push('> view# 为跳帧后 valid 序号（1 基），≠原始帧号。', '');
  const o = out.outliers.reproj;
  L.push(`- IQR（${IQR_K}×IQR）：${viewList(o.methods.iqr)}`);
  L.push(`- medSigma3（median±${SIGMA_K}σ）：${viewList(o.methods.medSigma3)}`);
  L.push(`- TOP${TOP_N}：${viewList(o.methods.top5)}`);
  L.push(`- 稳定离群（≥2 法命中）：${viewList(o.stable)}`, '');
  L.push('| view# | reprojL | reprojR | epi | 距离mm | 斜角° | 覆盖% | L-R比 | 双差 |', '|---|---|---|---|---|---|---|---|---|');
  for (const t of out.top5)
    L.push(`| ${t.view + 1} | ${fmt(t.reprojL)} | ${fmt(t.reprojR)} | ${fmt(t.epi)} | ${fmt(t.distL, 1)} | ${fmt(t.angleL, 1)} | ${fmt(t.coverL, 1)} | ${fmt(t.lrRatio, 2)} | ${t.bothBad == null ? 'N/A' : t.bothBad ? '是' : '否'} |`);
  L.push('');
  L.push('### 相关性', '');
  const pfmt = p => p == null ? 'N/A' : p < 1e-4 ? p.toExponential(2) : p.toFixed(4);
  const cfmt = c => c == null || c.r == null ? 'N/A' : `${fmt(c.r)}${c.sig}（p=${pfmt(c.p)}）`;
  L.push(`- corrDist（误差~距离（左目））：${cfmt(corr?.corrDist)}`);
  L.push(`- corrAngle（误差~斜角（左目））：${cfmt(corr?.corrAngle)}`);
  L.push(`- corrCoverInv（误差~覆盖取负（左目））：${cfmt(corr?.corrCoverInv)}`);
  L.push(`- corrFrameIdx（误差~帧序${corr?.coverage ? '（左目）' : '(最大侧)'}）：${cfmt(corr?.corrFrameIdx)}`);
  L.push(`- corrDistPartial（误差~距离 | 控制斜角）：${cfmt(corr?.corrDistPartial)}`);
  L.push(`- corrEdgePartial（误差~斜角 | 控制距离）：${cfmt(corr?.corrEdgePartial)}`);
  L.push(`- 偏度：L ${fmt(corr?.skew?.left)} / R ${fmt(corr?.skew?.right)} / max(L,R) ${fmt(corr?.skew?.max)}`);
  L.push(`- 峰度（超额）：L ${fmt(corr?.kurt?.left)} / R ${fmt(corr?.kurt?.right)} / max(L,R) ${fmt(corr?.kurt?.max)}`);
  L.push(`- fx-fy 差：L ${fmt(corr?.fxFyPct?.left, 3, '%')} / R ${fmt(corr?.fxFyPct?.right, 3, '%')}`);
  L.push(`- 可用样本 n：${corr?.n ?? 'N/A'}`);
  if (corr?.corrFrameIdx?.r != null && Math.abs(corr.corrFrameIdx.r) > 0.5)
    L.push('', '> ⚠ |corrFrameIdx| > 0.5：误差随采集顺序系统性变化，提示可能存在热漂移趋势，建议结合温度记录复核。');
  L.push('');
  const st = m._stats;
  L.push('### 标定可信度', '');
  const cov = st?.coverage;
  const covReason = cov?.frames?.reason ?? cov?.reason;
  if (covReason) {
    L.push(`- 帧数：N/A（${mdEsc(covReason)}）`);
    L.push(`- 姿态方向多样性：N/A（${mdEsc(covReason)}）`);
    L.push(`- 宫格覆盖：N/A（${mdEsc(covReason)}）`);
    L.push('- 综合判定：N/A');
  } else {
    L.push(`- 帧数：${cov?.frames?.value ?? 'N/A'}（≥${MIN_VALID_FRAMES} ${cov?.frames?.ok ? '✓' : '✗'}）`);
    L.push(`- 偏离首帧方向 >${ORIENT_ANGLE_DEG}° 的帧数：${cov?.orientationDiversity?.value ?? 'N/A'}（门槛 ≥1 ${cov?.orientationDiversity?.ok ? '✓' : '✗'}）`);
    const rc = cov?.regionCoverage?.value;
    if (rc) {
      L.push(`- 宫格帧数（3×3）：${rc.grid.map(row => row.join(' / ')).join(' ｜ ')}`);
      L.push(`- 四角空 ${rc.cornerEmpty} / 四边空 ${rc.edgeEmpty}（角空 ≤${CORNER_EMPTY_MAX} ${cov.regionCoverage.ok ? '✓' : '✗'}）`);
    } else L.push('- 宫格覆盖：N/A');
    L.push(`- 综合判定：${cov?.ok ? '✓ 通过' : '✗ 未过'}`);
  }
  L.push(`- 参考量级：良好制作下典型重投影误差约 ${REF_TYPICAL_REPROJ}px（${REF_REPROJ_SOURCE}）`, '');
  L.push('### 双目一致性', '');
  const ste = st?.stereo;
  if (!ste) L.push('- N/A');
  else {
    L.push(`- L-R 逐帧差：mean ${fmt(ste.lrDiff?.mean)} px / std ${fmt(ste.lrDiff?.std)} px`);
    L.push(`- corrLR（L~R）：${cfmt(ste.corrLR)}`);
    L.push(`- corrReprojEpi（max(L,R)~极线误差）：${cfmt(ste.corrReprojEpi)}`);
    const bfmt = b => b == null || b.beta == null ? 'N/A'
      : `β=${b.beta.toFixed(3)}${b.betaRobust == null ? '' : ` (robust ${b.betaRobust.toFixed(3)})`}`;
    const levWarn = b => b?.leverage != null && b.leverage > LEVERAGE_WARN;
    L.push(`- βL：${bfmt(ste.betaL)} / βR：${bfmt(ste.betaR)}${levWarn(ste.betaL) || levWarn(ste.betaR) ? ' ⚠杠杆敏感' : ''}`);
    const imb = ste.imbalancedFrames;
    L.push(`- 失衡帧（L/R 比 >${IMBALANCE_HI} 或 <${IMBALANCE_LO}）：左差 ${imb?.leftWorse ?? 'N/A'} / 右差 ${imb?.rightWorse ?? 'N/A'} / 共 ${imb?.count ?? 'N/A'} 帧${imb?.pct != null ? `（${imb.pct.toFixed(1)}%）` : ''}`);
  }
  L.push('### 共性问题指纹（规则库 v1）', '');
  L.push('| 规则 | 判定 | 证据摘要 |', '|---|---|---|');
  for (const r of m._fingerprints?.rules ?? [])
    L.push(`| ${r.id} ${mdEsc(r.name)} | ${FP_MARK[r.verdict] ?? r.verdict} | ${mdEsc(fpSummary(r))} |`);
  L.push('', '阈值为一期经验值（实证 n=4 批）', '');
  const fpR2 = m._fingerprints?.rules?.find(r => r.id === 'R2');
  if (fpR2?.verdict === 'hit' || cov == null || cov.ok == null) {
    L.push('### 物理检查清单（覆盖审计未过/不可测，请人工核查）', '',
      '- [ ] 标定板平整度与刚性（翘曲会降低精度——BoofCV/mrcal 均强调）',
      '- [ ] 双相机对焦与光圈一致性',
      '- [ ] 漫射均匀光照、无眩光无硬阴影',
      '- [ ] 棋盘格清洁无反光', '');
  }
  L.push('');
  return L.join('\n');
}
function renderProjectionDiff(pd) {
  const L = ['### 批次间投影差异（一致性指示器）', ''];
  for (const cmp of pd.comparisons ?? []) {
    L.push(`- ${mdEsc(cmp.base ?? '?')} → ${mdEsc(cmp.target ?? '?')}`, '');
    L.push('| 目 | maxPx | meanPx | kOnlyMaxPx |', '|---|---|---|---|');
    for (const [nm, s] of [['左目', cmp.left], ['右目', cmp.right]])
      L.push(`| ${nm} | ${fmt(s?.maxPx, 2)} | ${fmt(s?.meanPx, 2)} | ${fmt(s?.kOnlyMaxPx, 2)} |`);
    L.push('', '| 目 | Δfx% | Δcx px | Δcy px | Δk1 | Δk2 | Δk3 | Δp1 | Δp2 |', '|---|---|---|---|---|---|---|---|---|');
    for (const [nm, s] of [['左目', cmp.left], ['右目', cmp.right]]) {
      const d = s?.paramDelta;
      L.push(`| ${nm} | ${fmt(d?.dfxPct, 3)} | ${fmt(d?.dcx, 2)} | ${fmt(d?.dcy, 2)} | ${fmt(d?.dk1)} | ${fmt(d?.dk2)} | ${fmt(d?.dk3)} | ${fmt(d?.dp1)} | ${fmt(d?.dp2)} |`);
    }
    L.push('');
  }
  L.push('> 说明：该差异场无法区分镜头物理变化 / 拟合不稳定 / 参数漂移中的主导因素，仅作为跨批一致性指示器，不构成因果结论。', '');
  return L.join('\n');
}
function renderCompare(ms) {
  const L = ['## 跨批次对比', ''];
  L.push(`| 指标 | ${ms.map(m => mdEsc(m.label)).join(' | ')} |`);
  L.push(`|---|${ms.map(() => '---').join('|')}|`);
  L.push(`| verdict | ${ms.map(m => m.verdict).join(' | ')} |`);
  const base = ms[0];
  const mets = [
    ['reproj_mean (px)', m => m.reproj?.mean],
    ['epipolar_mean (px)', m => m.extrinsic?.epiMean],
    ['stereoRms (px)', m => m.extrinsic?.stereoRms],
    ['skipped%', m => m.frames?.skippedPct],
    ['baseline (mm)', m => m.baseline],
    ['focalDiff%', m => m.consistency?.focalDiffPct],
  ];
  for (const [nm, get] of mets) {
    L.push(`| ${nm} | ${ms.map(m => fmt(get(m))).join(' | ')} |`);
    L.push(`| Δ vs 首列 | ${ms.map(m => {
      const v = get(m), v0 = get(base);
      if (m === base || v == null || v0 == null || v0 === 0) return 'N/A';
      const d = (v - v0) / v0 * 100;
      return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`;
    }).join(' | ')} |`);
  }
  L.push(`| 稳定离群 (reproj) | ${ms.map(m => viewList(m._out?.outliers?.reproj?.stable)).join(' | ')} |`, '');
  L.push('### 高误差 view 重现', '');
  L.push('> 对齐假设：两批 valid 帧按采集序对齐，view# 相同不代表同一原始帧。', '');
  const p90of = m => quantiles((m._out?.rows ?? []).map(viewMaxErr).filter(Number.isFinite), .9);
  const p0 = p90of(base);
  for (const m of ms.slice(1)) {
    const p1 = p90of(m);
    const n = Math.min(base._out?.rows?.length ?? 0, m._out?.rows?.length ?? 0);
    const hits = [];
    if (p0 != null && p1 != null)
      for (let i = 0; i < n; i++) {
        const a = viewMaxErr(base._out.rows[i]), b = viewMaxErr(m._out.rows[i]);
        if (a != null && b != null && a > p0 && b > p1) hits.push(i + 1);
      }
    L.push(`- ${mdEsc(m.label)} vs ${mdEsc(base.label)}（对齐 ${n} 帧，P90 阈值 ${p0 != null ? p0.toFixed(4) : 'N/A'} / ${p1 != null ? p1.toFixed(4) : 'N/A'}）：${hits.length ? `view# ${hits.join(', ')}` : '无'}`);
  }
  L.push('');
  return L.join('\n');
}
function renderFingerprintRecurrence(rec) {
  const L = ['### 指纹跨批重现', ''];
  if (!rec?.length) L.push('无重现指纹', '');
  else {
    for (const r of rec) L.push(`- ${r.id}：${r.runs.map(mdEsc).join('、')}（${r.note}）`);
    L.push('');
  }
  return L.join('\n');
}
function analyzeAll(okRuns, args, outDir) {
  const ms = [];
  for (const r of okRuns) {
    const m = extractMetrics(r);
    if (m.fatal) { console.error(`[skip] ${m.path}: ${m.fatal}`); continue; }
    const th = resolveThresholds(args, m);
    gateMetrics(m, th);
    const cfg = loadBoardCfg(m);
    m.board = cfg;
    const geo = boardGeometry(m, cfg);
    m._out = analyzeOutliers(m, geo);
    m._corr = analyzeCorrelations(m._out, geo, m);
    m._stats = {
      stereo: stereoConsistency(m),
      coverage: coverageAudit(m, geo),
      referenceScale: { typicalReproj: REF_TYPICAL_REPROJ, source: REF_REPROJ_SOURCE },
    };
    m._th = th;
    m._fingerprints = matchFingerprints(m);
    ms.push(m);
  }
  if (!ms.length) { console.error('all inputs failed'); process.exit(2); }
  console.log('| label | verdict | reproj | epipolar | skipped% | baseline |');
  console.log('|---|---|---|---|---|---|');
  for (const m of ms)
    console.log(`| ${m.label} | ${m.verdict} | ${fmt(m.reproj?.mean)} | ${fmt(m.extrinsic?.epiMean)} | ${fmt(m.frames?.skippedPct, 1)} | ${fmt(m.baseline, 2)} |`);
  for (const m of ms) {
    const marks = [], clears = [];
    for (const r of m._fingerprints?.rules ?? []) {
      if (r.verdict === 'hit') marks.push(`${r.id}⚠`);
      else if (r.verdict === 'insufficient') marks.push(`${r.id}?`);
      else clears.push(r.id);
    }
    console.log(`指纹 ${m.label}: ${marks.length ? marks.join(' ') : '无命中'}${clears.length ? ` | 清单: ${clears.join(' ')}` : ''}`);
  }
  let dirOk = true;
  try { mkdirSync(outDir, { recursive: true }); }
  catch (e) { dirOk = false; console.error(`[warn] 输出目录创建失败：${e.message}`); }
  let chartsOk = !args.noCharts && dirOk;
  const chartsDir = join(outDir, 'charts');
  if (chartsOk) {
    try { mkdirSync(chartsDir, { recursive: true }); }
    catch (e) { chartsOk = false; console.error(`[warn] charts 目录创建失败，跳过全部图表：${e.message}`); }
  }
  if (chartsOk) {
    for (const [ri, m] of ms.entries()) {
      const files = {};
      for (const [k, svg] of Object.entries(buildCharts(m, m._out, m._corr, m.label))) {
        if (!svg) continue;
        const f = `${ri + 1}_${m.label.replace(/[^\w.-]/g, '_')}_${k}.svg`;
        try { writeFileSync(join(chartsDir, f), svg); files[k] = `charts/${f}`; }
        catch (e) { console.error(`[warn] 图表写失败，跳过 ${f}：${e.message}`); }
      }
      m._charts = files;
    }
  }
  const rep = ['# 相机标定分析报告', '', `- 生成时间：${now()}`, `- 输入：${ms.length} 个`, '', renderGlossary()];
  for (const m of ms) {
    rep.push(renderRunSection(m, m._out, m._corr, m._th));
    if (m._charts) for (const [k, p] of Object.entries(m._charts)) rep.push(`![${k}](${p})`);
    rep.push('');
  }
  let projectionDiff = null;
  let fingerprintRec = null;
  if (ms.length >= 2) {
    fingerprintRec = fingerprintRecurrence(ms);
    rep.push(renderCompare(ms));
    rep.push(renderFingerprintRecurrence(fingerprintRec));
    const comparisons = [];
    for (let i = 1; i < ms.length; i++)
      comparisons.push({
        base: ms[i - 1].label, target: ms[i].label,
        left: projectFieldDiff(ms[i - 1], ms[i], 'left'),
        right: projectFieldDiff(ms[i - 1], ms[i], 'right'),
      });
    projectionDiff = { nature: 'consistency-indicator', comparisons };
    rep.push(renderProjectionDiff(projectionDiff));
    if (chartsOk) {
      const cmpRows = [
        { metric: 'reproj_mean', items: ms.map(m => ({ label: m.label, value: m.reproj?.mean ?? null })) },
        { metric: 'epipolar_mean', items: ms.map(m => ({ label: m.label, value: m.extrinsic?.epiMean ?? null })) },
        { metric: 'skipped%', items: ms.map(m => ({ label: m.label, value: m.frames?.skippedPct ?? null })) },
      ];
      try {
        writeFileSync(join(chartsDir, 'compare.svg'), svgCompare('跨批次对比', cmpRows));
        rep.push('![compare](charts/compare.svg)', '');
      } catch (e) { console.error(`[warn] 图表写失败，跳过 compare.svg：${e.message}`); }
    }
  }
  let repOk = true;
  try {
    writeFileSync(join(outDir, 'report.md'), rep.join('\n') + '\n');
    console.log(`report: ${join(outDir, 'report.md')}`);
  } catch (e) { repOk = false; console.error(`[warn] report.md 写失败：${e.message}`); }
  const runsJson = ms.map(m => {
    const o = {};
    for (const [k, v] of Object.entries(m)) if (!k.startsWith('_')) o[k] = v;
    o.outliers = m._out.outliers;
    o.top5 = m._out.top5;
    o.correlations = m._corr;
    o.stats = m._stats;
    o.board = m.board;
    o.th = m._th;
    o.fingerprints = m._fingerprints.rules;
    if (m._charts) o.charts = m._charts;
    return o;
  });
  let sumOk = true;
  try {
    const summary = { generated: now(), inputCount: runsJson.length, runs: runsJson };
    if (projectionDiff) summary.projectionDiff = projectionDiff;
    if (fingerprintRec) summary.fingerprintRecurrence = fingerprintRec;
    writeFileSync(join(outDir, 'summary.json'), JSON.stringify(summary, null, 2));
    console.log(`summary: ${join(outDir, 'summary.json')}`);
  } catch (e) { sumOk = false; console.error(`[warn] summary.json 写失败：${e.message}`); }
  if (!repOk && !sumOk) process.exit(1);
}

const usage = 'usage: node analyze_camera_calib.mjs <camera_calib.json ...> [--out dir] [--reproj-th px] [--epipolar-th px] [--no-charts]';
const argv = process.argv.slice(2);
if (argv.includes('-h') || argv.includes('--help')) { console.log(usage); process.exit(0); }
if (argv.length === 0) { console.error(usage); process.exit(2); }
const args = parseArgs(argv);
if (args.errors.length) { console.error(args.errors.join('\n')); console.error(usage); process.exit(2); }
const usedLabels = new Set();
const runs = args.inputs.map(p => loadRun(p, usedLabels));
runs.forEach(r => r.fatal ? console.error(`[skip] ${r.path}: ${r.fatal}`) : console.log(`[load] ${r.label} <- ${r.path}`));
if (runs.every(r => r.fatal)) { console.error('all inputs failed'); process.exit(2); }
const okRuns = runs.filter(r => !r.fatal);
const outDir = args.out ?? defaultOutDir();
console.log(`[out ] ${outDir}`);
analyzeAll(okRuns, args, outDir);
