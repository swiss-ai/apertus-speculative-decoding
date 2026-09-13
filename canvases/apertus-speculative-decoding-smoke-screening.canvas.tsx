/**
 * Paired measurement of speculative decoding for Apertus-v1.5-70B: one baseline
 * deployment against two independently launched 8B draft deployments and one
 * model-free n-gram deployment.
 *
 * Every number rendered here is read from the committed run artifacts:
 *   - `results/smoke-screening-20260913-debug/analysis.csv` and each cell's
 *     `summary.json` (client-observed metrics, speculative counter deltas),
 *   - `metrics_after.prom` (`vllm:cache_config_info`) for KV-cache capacity,
 *   - `results/correctness/smoke-20260913-debug/` (exact-match comparisons and
 *     the calibrated divergence gate that replaced them),
 *   - `results/deployment-failures/draft-n3-tp1-3392110/` (unservable draft TP).
 * Derived quantities are labelled as derived. Nothing is extrapolated.
 */
import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";
import type { CalloutProps } from "cursor/canvas";

/** Renderable content, sourced from the SDK's own prop types. */
type Node = CalloutProps["children"];

/* ------------------------------------------------------------------ data -- */

type Arm = {
  ttftP50: number;
  ttftP95: number;
  tpotP50: number;
  tpotMean: number;
  tpotP95: number;
  outTps: number;
  e2eP50: number;
  e2eP95: number;
  gapP50: number;
  promptTokens: number;
  completionTokens: number;
};

type DraftArm = Arm & {
  drafts: number;
  draftTokens: number;
  acceptedTokens: number;
  acceptanceRate: number;
  meanAcceptanceLength: number;
  perPosition: [number, number, number];
};

type Cell = {
  key: string;
  short: string;
  workload: string;
  concurrency: number;
  baseline: Arm;
  /** `draft-n3-tp4` deployment repeat 1, job 3391426 on nid006633. */
  draft1: DraftArm;
  /** `draft-n3-tp4` deployment repeat 2, job 3392153 on nid006687. */
  draft2: DraftArm;
  /** `ngram-n3` (prompt lookup 1–4), job 3392370 on nid006687. */
  ngram: DraftArm;
  /** Manual label nudge for the scatter plot, in px. */
  labelDx: number;
  labelDy: number;
};

const CELLS: Cell[] = [
  {
    key: "chat-c1",
    short: "chat c1",
    workload: "open_chat",
    concurrency: 1,
    labelDx: 9,
    labelDy: -9,
    baseline: {
      ttftP50: 31.01577004417777,
      ttftP95: 35.53785252152011,
      tpotP50: 14.129014411718456,
      tpotMean: 14.130165289924436,
      tpotP95: 14.138498527170963,
      outTps: 70.12972149602668,
      e2eP50: 1924.3120565079153,
      e2eP95: 2735.9838715055957,
      gapP50: 14.11726651713252,
      promptTokens: 1944,
      completionTokens: 3240,
    },
    draft1: {
      ttftP50: 111.62124352995306,
      ttftP95: 113.47290540579706,
      tpotP50: 31.864067355517847,
      tpotMean: 31.836271620789336,
      tpotP95: 32.49081057331325,
      outTps: 31.0126732695681,
      e2eP50: 4367.674916982651,
      e2eP95: 6133.702770061791,
      gapP50: 90.76851792633533,
      promptTokens: 1944,
      completionTokens: 3264,
      drafts: 1128,
      draftTokens: 3384,
      acceptedTokens: 2124,
      acceptanceRate: 0.6276595744680851,
      meanAcceptanceLength: 2.882978723404255,
      perPosition: [0.7659574468085106, 0.6382978723404256, 0.4787234042553192],
    },
    draft2: {
      ttftP50: 120.22672547027469,
      ttftP95: 122.23579798592255,
      tpotP50: 34.172717401705725,
      tpotMean: 34.21347855522761,
      tpotP95: 35.79275011819751,
      outTps: 29.092050925791096,
      e2eP50: 4636.392372543924,
      e2eP95: 6501.595394266769,
      gapP50: 96.86424140818417,
      promptTokens: 1944,
      completionTokens: 3240,
      drafts: 1116,
      draftTokens: 3348,
      acceptedTokens: 2136,
      acceptanceRate: 0.6379928315412187,
      meanAcceptanceLength: 2.913978494623656,
      perPosition: [0.8064516129032258, 0.6559139784946236, 0.45161290322580644],
    },
    ngram: {
      ttftP50: 24.12557799834758,
      ttftP95: 24.604766292031854,
      tpotP50: 16.024629588666592,
      tpotMean: 16.013892850515003,
      tpotP95: 16.688393220432445,
      outTps: 63.25262857834419,
      e2eP50: 2134.570181486197,
      e2eP95: 2974.927077477332,
      gapP50: 18.542225006967783,
      promptTokens: 1944,
      completionTokens: 3240,
      drafts: 1092,
      draftTokens: 3276,
      acceptedTokens: 456,
      acceptanceRate: 0.1391941391941392,
      meanAcceptanceLength: 1.4175824175824177,
      perPosition: [0.2087912087912088, 0.13186813186813187, 0.07692307692307693],
    },
  },
  {
    key: "chat-c8",
    short: "chat c8",
    workload: "open_chat",
    concurrency: 8,
    labelDx: 9,
    labelDy: 15,
    baseline: {
      ttftP50: 44.3590774666518,
      ttftP95: 64.93447046959773,
      tpotP50: 14.69453933682549,
      tpotMean: 14.670868443973172,
      tpotP95: 14.714398575106081,
      outTps: 432.25749510362783,
      e2eP50: 1997.5567575311288,
      e2eP95: 2870.039315603208,
      gapP50: 14.661136432550848,
      promptTokens: 1944,
      completionTokens: 3240,
    },
    draft1: {
      ttftP50: 197.11150752846152,
      ttftP95: 221.91237719962373,
      tpotP50: 32.54491677104559,
      tpotMean: 32.409641558628444,
      tpotP95: 33.797205998652984,
      outTps: 190.5939564277659,
      e2eP50: 4400.5185349378735,
      e2eP95: 6509.723449638113,
      gapP50: 91.21225692797452,
      promptTokens: 1944,
      completionTokens: 3248,
      drafts: 1122,
      draftTokens: 3366,
      acceptedTokens: 2131,
      acceptanceRate: 0.6330956625074272,
      meanAcceptanceLength: 2.8992869875222818,
      perPosition: [0.7976827094474154, 0.6550802139037433, 0.446524064171123],
    },
    draft2: {
      ttftP50: 233.69901347905397,
      ttftP95: 248.0323146446608,
      tpotP50: 35.54313675707494,
      tpotMean: 35.133413964303486,
      tpotP95: 36.56369111479514,
      outTps: 173.19112511150914,
      e2eP50: 4784.969286993146,
      e2eP95: 6922.576771967579,
      gapP50: 98.08018407784402,
      promptTokens: 1944,
      completionTokens: 3240,
      drafts: 1123,
      draftTokens: 3369,
      acceptedTokens: 2116,
      acceptanceRate: 0.6280795488275452,
      meanAcceptanceLength: 2.8842386464826357,
      perPosition: [0.7809439002671416, 0.6438112199465716, 0.4594835262689225],
    },
    ngram: {
      ttftP50: 39.53329741489142,
      ttftP95: 48.82720810128376,
      tpotP50: 17.359798584163656,
      tpotMean: 17.213493244944882,
      tpotP95: 18.245461702878984,
      outTps: 366.5164195977573,
      e2eP50: 2275.099836057052,
      e2eP95: 3236.614705028478,
      gapP50: 19.586749840527773,
      promptTokens: 1944,
      completionTokens: 3240,
      drafts: 1075,
      draftTokens: 3225,
      acceptedTokens: 452,
      acceptanceRate: 0.14015503875968993,
      meanAcceptanceLength: 1.4204651162790698,
      perPosition: [0.21395348837209302, 0.13116279069767442, 0.07534883720930233],
    },
  },
  {
    key: "code-c1",
    short: "code c1",
    workload: "code",
    concurrency: 1,
    labelDx: 9,
    labelDy: -9,
    baseline: {
      ttftP50: 27.751603513024747,
      ttftP95: 30.49125950783491,
      tpotP50: 14.150161672427316,
      tpotMean: 14.15406258929674,
      tpotP95: 14.166318852297582,
      outTps: 70.46974580539374,
      e2eP50: 5447.2582885064185,
      e2eP95: 5456.618541537318,
      gapP50: 14.14400152862072,
      promptTokens: 2184,
      completionTokens: 9216,
    },
    draft1: {
      ttftP50: 131.12756691407412,
      ttftP95: 133.49370927317068,
      tpotP50: 30.160477702255427,
      tpotMean: 30.037497874865878,
      tpotP95: 31.152575448892193,
      outTps: 33.01537628400447,
      e2eP50: 11684.548205928877,
      e2eP95: 12051.120656274725,
      gapP50: 96.04675450827926,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 2748,
      draftTokens: 8244,
      acceptedTokens: 6492,
      acceptanceRate: 0.7874818049490538,
      meanAcceptanceLength: 3.3624454148471616,
      perPosition: [0.8864628820960698, 0.7729257641921398, 0.7030567685589519],
    },
    draft2: {
      ttftP50: 128.0825735302642,
      ttftP95: 140.3792893863283,
      tpotP50: 30.543017312981597,
      tpotMean: 30.54171125447597,
      tpotP95: 32.41316932469349,
      outTps: 32.46565710703983,
      e2eP50: 11822.260092943907,
      e2eP95: 12541.750745859463,
      gapP50: 99.51842087320983,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 2724,
      draftTokens: 8172,
      acceptedTokens: 6516,
      acceptanceRate: 0.7973568281938326,
      meanAcceptanceLength: 3.392070484581498,
      perPosition: [0.9074889867841409, 0.788546255506608, 0.6960352422907489],
    },
    ngram: {
      ttftP50: 26.481231092475355,
      ttftP95: 27.485086291562762,
      tpotP50: 14.906808968594508,
      tpotMean: 15.016869529173931,
      tpotP95: 16.291230336450496,
      outTps: 66.45707028221716,
      e2eP50: 5735.818793531507,
      e2eP95: 6267.282484599855,
      gapP50: 17.141888034529984,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 4164,
      draftTokens: 12444,
      acceptedTokens: 1584,
      acceptanceRate: 0.12729026036644167,
      meanAcceptanceLength: 1.3804034582132565,
      perPosition: [0.21613832853025935, 0.09798270893371758, 0.06628242074927954],
    },
  },
  {
    key: "code-c8",
    short: "code c8",
    workload: "code",
    concurrency: 8,
    labelDx: 9,
    labelDy: 16,
    baseline: {
      ttftP50: 44.78542983997613,
      ttftP95: 66.70471947873011,
      tpotP50: 14.700184079548514,
      tpotMean: 14.700670462028546,
      tpotP95: 14.706115846099046,
      outTps: 540.6142187291132,
      e2eP50: 5676.359422970563,
      e2eP95: 5696.285984723363,
      gapP50: 14.69302293844521,
      promptTokens: 2184,
      completionTokens: 9216,
    },
    draft1: {
      ttftP50: 218.02396804559976,
      ttftP95: 251.06144183082506,
      tpotP50: 30.069864900933467,
      tpotMean: 29.97578495724963,
      tpotP95: 30.942968094105012,
      outTps: 255.17896653041808,
      e2eP50: 11733.487372519448,
      e2eP95: 12083.562990487553,
      gapP50: 96.22802282683551,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 2753,
      draftTokens: 8259,
      acceptedTokens: 6490,
      acceptanceRate: 0.7858094200266376,
      meanAcceptanceLength: 3.357428260079913,
      perPosition: [0.891754449691246, 0.7686160552124954, 0.6970577551761714],
    },
    draft2: {
      ttftP50: 245.59995508752763,
      ttftP95: 273.288237163797,
      tpotP50: 31.74796237825482,
      tpotMean: 31.65958549735036,
      tpotP95: 33.35210565841142,
      outTps: 238.47738833740593,
      e2eP50: 12424.783781403676,
      e2eP95: 13027.731159562245,
      gapP50: 101.11371893435717,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 2765,
      draftTokens: 8295,
      acceptedTokens: 6475,
      acceptanceRate: 0.7805907172995781,
      meanAcceptanceLength: 3.3417721518987342,
      perPosition: [0.8817359855334539, 0.7641952983725135, 0.6958408679927667],
    },
    ngram: {
      ttftP50: 39.42043904680759,
      ttftP95: 55.18617081688717,
      tpotP50: 16.042244911203575,
      tpotMean: 16.09481726118057,
      tpotP95: 17.519956833030054,
      outTps: 464.29014988769256,
      e2eP50: 6183.537346543744,
      e2eP95: 6749.206578801386,
      gapP50: 19.51587398070842,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 4119,
      draftTokens: 12305,
      acceptedTokens: 1623,
      acceptanceRate: 0.13189760260056888,
      meanAcceptanceLength: 1.394027676620539,
      perPosition: [0.2231124059237679, 0.10390871570769604, 0.06700655498907501],
    },
  },
  {
    key: "summ-c1",
    short: "summ c1",
    workload: "long_context_summarization",
    concurrency: 1,
    labelDx: 9,
    labelDy: 16,
    baseline: {
      ttftP50: 33.5973211331293,
      ttftP95: 36.52248587459326,
      tpotP50: 14.158176855179724,
      tpotMean: 14.163059087760203,
      tpotP95: 14.176949262596949,
      outTps: 70.21602782996129,
      e2eP50: 3332.336171530187,
      e2eP95: 3651.232915208675,
      gapP50: 14.147759531624615,
      promptTokens: 6780,
      completionTokens: 5616,
    },
    draft1: {
      ttftP50: 120.28303404804319,
      ttftP95: 132.50504455063492,
      tpotP50: 32.720212998670405,
      tpotMean: 33.38591971027642,
      tpotP95: 36.782325995519905,
      outTps: 29.42025630021274,
      e2eP50: 7724.188559921458,
      e2eP95: 9501.645165681839,
      gapP50: 93.5299025150016,
      promptTokens: 6780,
      completionTokens: 5592,
      drafts: 1932,
      draftTokens: 5796,
      acceptedTokens: 3660,
      acceptanceRate: 0.6314699792960663,
      meanAcceptanceLength: 2.8944099378881987,
      perPosition: [0.7639751552795031, 0.6211180124223602, 0.5093167701863354],
    },
    draft2: {
      ttftP50: 130.19031891599298,
      ttftP95: 142.6385407568887,
      tpotP50: 37.386853766620376,
      tpotMean: 37.34989499394505,
      tpotP95: 39.89651821069784,
      outTps: 26.33708601511989,
      e2eP50: 8899.068581988104,
      e2eP95: 10312.37978542922,
      gapP50: 101.27478453796357,
      promptTokens: 6780,
      completionTokens: 5628,
      drafts: 2004,
      draftTokens: 6012,
      acceptedTokens: 3636,
      acceptanceRate: 0.6047904191616766,
      meanAcceptanceLength: 2.81437125748503,
      perPosition: [0.7604790419161677, 0.5808383233532934, 0.47305389221556887],
    },
    ngram: {
      ttftP50: 27.737956028431654,
      ttftP95: 28.394636802840978,
      tpotP50: 12.511545461975917,
      tpotMean: 12.514326372803334,
      tpotP95: 13.099275814807589,
      outTps: 79.16688596452263,
      e2eP50: 2959.5145795028657,
      e2eP95: 3368.55119203683,
      gapP50: 17.081297119148076,
      promptTokens: 6780,
      completionTokens: 5628,
      drafts: 2244,
      draftTokens: 6732,
      acceptedTokens: 1668,
      acceptanceRate: 0.24777183600713013,
      meanAcceptanceLength: 1.7433155080213902,
      perPosition: [0.31016042780748665, 0.25133689839572193, 0.18181818181818182],
    },
  },
  {
    key: "summ-c8",
    short: "summ c8",
    workload: "long_context_summarization",
    concurrency: 8,
    labelDx: -64,
    labelDy: 4,
    baseline: {
      ttftP50: 45.67884199786931,
      ttftP95: 70.48749225214124,
      tpotP50: 14.716388219439306,
      tpotMean: 14.707432902066513,
      tpotP95: 14.732173119416377,
      outTps: 493.6571058686475,
      e2eP50: 3471.9936264446005,
      e2eP95: 3822.6895610103384,
      gapP50: 14.703823020681739,
      promptTokens: 6780,
      completionTokens: 5620,
    },
    draft1: {
      ttftP50: 198.44826601911336,
      ttftP95: 234.4543559011072,
      tpotP50: 31.36211582968766,
      tpotMean: 32.04707510297599,
      tpotP95: 34.18852263153074,
      outTps: 211.96595167755117,
      e2eP50: 7554.02875400614,
      e2eP95: 8952.794521988835,
      gapP50: 91.05295315384865,
      promptTokens: 6780,
      completionTokens: 5624,
      drafts: 1967,
      draftTokens: 5901,
      acceptedTokens: 3660,
      acceptanceRate: 0.6202338586680224,
      meanAcceptanceLength: 2.8607015760040673,
      perPosition: [0.7661413319776309, 0.6014234875444839, 0.4931367564819522],
    },
    draft2: {
      ttftP50: 214.80390650685877,
      ttftP95: 256.56637334031984,
      tpotP50: 33.097804971771026,
      tpotMean: 33.89149068103166,
      tpotP95: 36.89481359043652,
      outTps: 200.80871487361844,
      e2eP50: 7977.253047982231,
      e2eP95: 9664.550899865571,
      gapP50: 97.34926605597138,
      promptTokens: 6780,
      completionTokens: 5624,
      drafts: 1942,
      draftTokens: 5826,
      acceptedTokens: 3690,
      acceptanceRate: 0.6333676622039135,
      meanAcceptanceLength: 2.9001029866117403,
      perPosition: [0.7811534500514933, 0.6225540679711637, 0.4963954685890834],
    },
    ngram: {
      ttftP50: 41.04452847968787,
      ttftP95: 57.13180947350338,
      tpotP50: 13.36328009518559,
      tpotMean: 13.356094826303861,
      tpotP95: 14.346185312652047,
      outTps: 528.7114538584626,
      e2eP50: 3281.3372125383466,
      e2eP95: 3699.1192806512117,
      gapP50: 19.533600541763008,
      promptTokens: 6780,
      completionTokens: 5644,
      drafts: 2216,
      draftTokens: 6648,
      acceptedTokens: 1763,
      acceptanceRate: 0.26519253910950663,
      meanAcceptanceLength: 1.7955776173285198,
      perPosition: [0.34205776173285196, 0.26353790613718414, 0.18998194945848376],
    },
  },
];

/** `vllm:cache_config_info` from each deployment's `metrics_after.prom`. */
const KV_CACHE = {
  baseline: { tokens: 513696, maxConcurrency: 3.919189453125, gpuBlocks: 32106 },
  ngram: { tokens: 416640, maxConcurrency: 3.1787109375, gpuBlocks: 26040 },
  draft1: { tokens: 251472, maxConcurrency: 1.9185791015625, gpuBlocks: 15717 },
  draft2: { tokens: 251488, maxConcurrency: 1.918701171875, gpuBlocks: 15718 },
};

/** `configs/experiment.yaml`, `validity_gates.maximum_repeat_spread_fraction`. */
const MAX_REPEAT_SPREAD = 0.1;

/**
 * `results/correctness/smoke-20260913-debug/`. Exact-match counts come from the
 * original `comparison*.json`; divergence statistics and verdicts from the
 * `divergence-*.json` and `gate-*.json` re-analysis of the same captures.
 */
const CORRECTNESS = {
  pairs: [
    {
      key: "b-d1",
      label: "baseline vs. draft repeat 1",
      role: "compared pair",
      exactMatches: 0,
      worstPrefixCharacters: 297,
      worstPrefixFraction: 0.18390092879256967,
      meanEditDistance: 0.08112920838220665,
      worstEditDistance: 0.22652885443583118,
      worstTokenDifference: 3,
      verdict: "within_control_envelope",
    },
    {
      key: "b-d2",
      label: "baseline vs. draft repeat 2",
      role: "compared pair",
      exactMatches: 3,
      worstPrefixCharacters: 151,
      worstPrefixFraction: 0.12541528239202657,
      meanEditDistance: 0.10416822997550906,
      worstEditDistance: 0.26151315789473684,
      worstTokenDifference: 0,
      verdict: "within_control_envelope",
    },
    {
      key: "d1-d2",
      label: "draft repeat 1 vs. draft repeat 2",
      role: "calibration control",
      exactMatches: 0,
      worstPrefixCharacters: 151,
      worstPrefixFraction: 0.12818336162988114,
      meanEditDistance: 0.1469385371130544,
      worstEditDistance: 0.2558139534883721,
      worstTokenDifference: 3,
      verdict: "control",
    },
    {
      key: "b-ng",
      label: "baseline vs. n-gram",
      role: "no control exists",
      exactMatches: 2,
      worstPrefixCharacters: 151,
      worstPrefixFraction: 0.12731871838111297,
      meanEditDistance: 0.11668092393051567,
      worstEditDistance: 0.3082865168539326,
      worstTokenDifference: 2,
      verdict: "not judged",
    },
  ],
  totalCases: 6,
  allowance: {
    prefixFraction: 0.07818336162988114,
    editDistance: 0.3058139534883721,
    tokenDifference: 5,
  },
  margins: { prefixFraction: 0.05, editDistance: 0.05, tokenDifference: 2 },
};

/* --------------------------------------------------------------- derived -- */

const CATEGORIES = CELLS.map((c) => c.short);

const METRICS = {
  ttftP50: { label: "TTFT p50", unit: "ms", axis: "Time to first token, p50 (ms)" },
  ttftP95: { label: "TTFT p95", unit: "ms", axis: "Time to first token, p95 (ms)" },
  tpotP50: { label: "TPOT p50", unit: "ms", axis: "Time per output token, p50 (ms)" },
  tpotP95: { label: "TPOT p95", unit: "ms", axis: "Time per output token, p95 (ms)" },
  outTps: {
    label: "Output throughput",
    unit: "tok/s",
    axis: "Output throughput (completion tokens / s)",
  },
} as const;

type MetricId = keyof typeof METRICS;

const METRIC_ORDER: MetricId[] = ["ttftP50", "ttftP95", "tpotP50", "tpotP95", "outTps"];

const num = (v: number, digits = 2) =>
  v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const pct = (v: number, digits = 1) => `${num(v * 100, digits)}%`;
const ratio = (a: number, b: number) => a / b;
const round = (v: number, digits = 2) => Number(v.toFixed(digits));
/** Absolute difference over the mean of the two repeats, as in the run README. */
const spread = (a: number, b: number) => Math.abs(a - b) / ((a + b) / 2);

const draftArms = (cell: Cell): [DraftArm, DraftArm] => [cell.draft1, cell.draft2];

const throughputRatios1 = CELLS.map((c) => ratio(c.draft1.outTps, c.baseline.outTps));
const throughputRatios2 = CELLS.map((c) => ratio(c.draft2.outTps, c.baseline.outTps));
const allThroughputRatios = [...throughputRatios1, ...throughputRatios2];
const allTpotRatios = CELLS.flatMap((c) =>
  draftArms(c).map((arm) => ratio(arm.tpotP50, c.baseline.tpotP50)),
);
const allTtftRatios = CELLS.flatMap((c) =>
  draftArms(c).map((arm) => ratio(arm.ttftP50, c.baseline.ttftP50)),
);
const allAcceptances = CELLS.flatMap((c) => draftArms(c).map((arm) => arm.acceptanceRate));
const allMals = CELLS.flatMap((c) => draftArms(c).map((arm) => arm.meanAcceptanceLength));

const ngramRatios = CELLS.map((c) => ratio(c.ngram.outTps, c.baseline.outTps));
const ngramAcceptances = CELLS.map((c) => c.ngram.acceptanceRate);
const ngramTtftRatios = CELLS.map((c) => ratio(c.ngram.ttftP50, c.baseline.ttftP50));
const ngramOverDraft = CELLS.map((c) => ratio(c.ngram.outTps, c.draft1.outTps));
const winningCells = CELLS.filter((_, i) => ngramRatios[i] > 1);

/**
 * Derived step time: mean TPOT × T ÷ (T − accepted), where T is the cell's
 * completion tokens. Every accepted draft token is a token that needed no step
 * of its own, so T − accepted is the number of engine steps. The baseline emits
 * one token per step, so its step time is its TPOT.
 */
const stepMs = (arm: Arm, accepted = 0) =>
  (arm.tpotMean * arm.completionTokens) / (arm.completionTokens - accepted);
const baselineSteps = CELLS.map((c) => stepMs(c.baseline));
const ngramSteps = CELLS.map((c) => stepMs(c.ngram, c.ngram.acceptedTokens));
const draft1Steps = CELLS.map((c) => stepMs(c.draft1, c.draft1.acceptedTokens));
const draft2Steps = CELLS.map((c) => stepMs(c.draft2, c.draft2.acceptedTokens));
/** Verification plus the scheduling change, in baseline decode steps. */
const verifyCost = CELLS.map((_, i) => ngramSteps[i] / baselineSteps[i]);
/** The 8B drafter's additional cost, in baseline decode steps, over both repeats. */
const draftCost = [...draft1Steps, ...draft2Steps].map(
  (step, i) => (step - ngramSteps[i % CELLS.length]) / baselineSteps[i % CELLS.length],
);
const draftStepCost = [...draft1Steps, ...draft2Steps].map(
  (step, i) => step / baselineSteps[i % CELLS.length],
);

const acceptanceSpreads = CELLS.map((c) => spread(c.draft1.acceptanceRate, c.draft2.acceptanceRate));
const ratioSpreads = CELLS.map((_, i) => spread(throughputRatios1[i], throughputRatios2[i]));
const breachingCells = CELLS.filter((_, i) => ratioSpreads[i] > MAX_REPEAT_SPREAD);

const span = (xs: number[], digits = 2, suffix = "") =>
  `${num(Math.min(...xs), digits)}–${num(Math.max(...xs), digits)}${suffix}`;
const spanPct = (xs: number[], digits = 0) =>
  `${num(Math.min(...xs) * 100, digits)}–${num(Math.max(...xs) * 100, digits)}%`;

/* ---------------------------------------------------------------- pieces -- */

function Caption({ children }: { children: Node }) {
  return (
    <Text size="small" tone="tertiary">
      {children}
    </Text>
  );
}

function Header() {
  return (
    <Stack gap={10}>
      <H1>Speculative decoding on Apertus-v1.5-70B: draft model against n-gram</H1>
      <Text tone="secondary">
        One 4×GH200 Clariden node per deployment. Baseline `swiss-ai/Apertus-v1.5-70B` at
        tensor-parallel 4; an `Apertus-v1.5-8B` draft model at `num_speculative_tokens=3`,
        `draft_tensor_parallel_size=4`, launched twice independently; and model-free n-gram
        speculation at depth 3 with prompt lookup 1–4. Three workloads × two concurrencies × four
        deployments = 24 cells, 8 warmup then 24 measured requests each, all with a 1.000 success
        rate.
      </Text>
      <Row gap={6} wrap>
        <Pill size="sm">Clariden · partition debug</Pill>
        <Pill size="sm">2026-09-13, 15:35–18:15 UTC</Pill>
        <Pill size="sm">jobs 3391425 / 3391426 / 3392153 / 3392370</Pill>
        <Pill size="sm">smoke corpus, 6 prompts</Pill>
        <Pill size="sm" tone="warning">
          2 draft repeats, 1 baseline
        </Pill>
        <Pill size="sm" tone="warning">
          draft TP=1 unservable
        </Pill>
      </Row>
    </Stack>
  );
}

function Headline() {
  return (
    <Grid columns="1.5fr 1fr" gap={20} align="start">
      <Stack gap={16}>
        <Text weight="semibold" style={{ fontSize: 17, lineHeight: 1.45 }}>
          The 8B drafter's tokens are accepted {spanPct(allAcceptances)} of the time and it loses in all
          twelve of its cells. N-gram accepts only {spanPct(ngramAcceptances)} and is the only arm that
          wins anything, taking {span(ngramRatios, 3)}× of baseline throughput and beating the baseline
          outright on long-context summarization. Acceptance is not the binding constraint — the arm
          that accepts worst is the arm that wins.
        </Text>
        <Grid columns={3} gap={16}>
          <Stat
            value={`${span(allThroughputRatios)}×`}
            label="draft-n3-tp4 output throughput vs baseline (12 cells)"
            tone="danger"
          />
          <Stat
            value={`${span(ngramRatios)}×`}
            label="ngram-n3 output throughput vs baseline (6 cells)"
            tone="success"
          />
          <Stat
            value={`${span(ngramOverDraft, 2)}×`}
            label="ngram-n3 over draft-n3-tp4 repeat 1"
            tone="success"
          />
          <Stat value={`${span(allTpotRatios)}×`} label="draft TPOT p50 vs baseline" tone="danger" />
          <Stat
            value={`${span(draftStepCost, 1)}×`}
            label="draft engine step cost vs baseline decode step (derived)"
            tone="danger"
          />
          <Stat
            value={`−${num((1 - KV_CACHE.draft1.tokens / KV_CACHE.baseline.tokens) * 100, 1)}%`}
            label="KV-cache tokens after hosting the drafter"
            tone="danger"
          />
        </Grid>
      </Stack>
      <Card>
        <CardHeader trailing="held constant">configuration</CardHeader>
        <CardBody>
          <Stack gap={6}>
            {(
              [
                ["target", "swiss-ai/Apertus-v1.5-70B, TP=4"],
                ["draft", "swiss-ai/Apertus-v1.5-8B, TP=4"],
                ["n-gram", "prompt lookup 1–4, no drafter"],
                ["speculative tokens", "3"],
                ["max_model_len", "131072"],
                ["gpu_memory_utilization", "0.8"],
                ["sampling", "temperature 0.0, top_p 1.0, natural EOS"],
                ["per cell", "8 warmup + 24 measured, closed loop"],
                ["load generator", "on-cluster, direct to replica node IP"],
              ] as Array<[string, string]>
            ).map(([k, v]) => (
              <Row key={k} gap={10} align="start" justify="space-between">
                <Text size="small" tone="tertiary">
                  {k}
                </Text>
                <Text size="small" style={{ textAlign: "right" }}>
                  {v}
                </Text>
              </Row>
            ))}
          </Stack>
        </CardBody>
      </Card>
    </Grid>
  );
}

function DeploymentSpread() {
  const breach = breachingCells[0];
  const breachIndex = CELLS.findIndex((c) => c.key === breach.key);
  return (
    <Stack gap={12}>
      <H2>What reproduces between deployments, and what does not</H2>
      <Text tone="secondary">
        Jobs 3391426 and 3392153 served the identical configuration on different nodes at different
        times. The protocol's independent unit is the deployment repeat, so the gap between these two
        is the measurement's own noise floor — and it is not the same size for every quantity.
        Acceptance moves by {span(acceptanceSpreads.map((s) => s * 100), 1)}%, the throughput ratio by{" "}
        {span(ratioSpreads.map((s) => s * 100), 1)}%. The drafter's acceptance is a property of the
        model pair; the speedup is a property of the deployment.
      </Text>
      <Grid columns="1.4fr 1fr" gap={20} align="start">
        <Stack gap={8}>
          <BarChart
            categories={CATEGORIES}
            series={[
              {
                name: "Acceptance rate",
                data: acceptanceSpreads.map((s) => round(s * 100, 1)),
                tone: "success",
              },
              {
                name: "Output-throughput ratio vs baseline",
                data: ratioSpreads.map((s) => round(s * 100, 1)),
                tone: "danger",
              },
            ]}
            valueSuffix="%"
            height={280}
          />
          <Caption>
            y: between-deployment spread, |repeat 1 − repeat 2| ÷ mean of the two, in percent · x:
            workload × concurrency cell · lower is more reproducible · the protocol's
            `maximum_repeat_spread_fraction` is {pct(MAX_REPEAT_SPREAD, 0)}, which only the throughput
            ratio approaches, and only `{breach.workload}` at concurrency {breach.concurrency}{" "}
            exceeds, at {pct(ratioSpreads[breachIndex], 1)} · source: per-cell `summary.json` in both
            draft deployments.
          </Caption>
        </Stack>
        <Stack gap={12}>
          <Callout
            tone="warning"
            title={`${breach.workload} at concurrency ${breach.concurrency} breaches the repeat-spread gate`}
          >
            <Text size="small">
              Its throughput ratio moves {pct(ratioSpreads[breachIndex], 1)} between the two
              deployments, against a declared limit of {pct(MAX_REPEAT_SPREAD, 0)}. Treat that cell's
              effect size as the least settled of the six; its direction matches the others. The
              second-largest, `open_chat` at concurrency 8, sits just under the limit at{" "}
              {pct(ratioSpreads[CELLS.findIndex((c) => c.key === "chat-c8")], 1)}.
            </Text>
          </Callout>
          <Callout tone="warning" title="Repeat 2 is slower in all six cells">
            <Text size="small">
              A symmetric noise process would put roughly half the cells on each side. All six moving
              the same way is the signature of a node or drift effect, which is exactly what two
              repeats cannot separate — and a reason not to read small differences between them as
              anything else.
            </Text>
          </Callout>
        </Stack>
      </Grid>
      <Table
        headers={[
          "Workload",
          "Conc.",
          "Acceptance r1 → r2",
          "Acceptance spread",
          "Output tok/s r1 → r2",
          "Ratio to baseline r1 → r2",
          "Ratio spread",
        ]}
        columnAlign={["left", "right", "right", "right", "right", "right", "right"]}
        rowTone={CELLS.map((_, i) => (ratioSpreads[i] > MAX_REPEAT_SPREAD ? "danger" : undefined))}
        rows={CELLS.map((c, i) => [
          c.workload,
          c.concurrency,
          `${num(c.draft1.acceptanceRate, 3)} → ${num(c.draft2.acceptanceRate, 3)}`,
          pct(acceptanceSpreads[i], 1),
          `${num(c.draft1.outTps, 1)} → ${num(c.draft2.outTps, 1)}`,
          `${num(throughputRatios1[i], 3)} → ${num(throughputRatios2[i], 3)}`,
          pct(ratioSpreads[i], 1),
        ])}
        striped
      />
      <Caption>
        Both draft deployments ran `num_speculative_tokens=3`, `draft_tensor_parallel_size=4`, the
        same corpus, request counts, warmup, sampling, and load path. Repeat 1 was driven by
        `apertus-bench matrix`, repeat 2 by `apertus-bench run` per cell. Two repeats support a spread
        but not a bootstrap confidence interval over deployment effects, and the baseline still has
        only one deployment, so every ratio on this canvas inherits the uncertainty of a single
        baseline.
      </Caption>
    </Stack>
  );
}

function PairedComparison() {
  const [metricId, setMetricId] = useCanvasState<MetricId>("pairedMetric", "tpotP50");
  const metric = METRICS[metricId];
  const higherIsWorse = metricId !== "outTps";

  return (
    <Stack gap={12}>
      <H2>Paired per-cell comparison</H2>
      <Text tone="secondary">
        The same corpus, request order, and client drive every deployment. Pick a metric; the
        regression has the same sign in every workload/concurrency cell and in both draft
        deployments.
      </Text>
      <Row gap={6} wrap>
        {METRIC_ORDER.map((id) => (
          <Pill key={id} active={id === metricId} onClick={() => setMetricId(id)}>
            {METRICS[id].label}
          </Pill>
        ))}
      </Row>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "baseline, job 3391425 (method=none)",
            data: CELLS.map((c) => round(c.baseline[metricId], 2)),
            tone: "info",
          },
          {
            name: "draft-n3-tp4 repeat 1, job 3391426",
            data: CELLS.map((c) => round(c.draft1[metricId], 2)),
            tone: "danger",
          },
          {
            name: "draft-n3-tp4 repeat 2, job 3392153",
            data: CELLS.map((c) => round(c.draft2[metricId], 2)),
            tone: "warning",
          },
          {
            name: "ngram-n3, job 3392370",
            data: CELLS.map((c) => round(c.ngram[metricId], 2)),
            tone: "success",
          },
        ]}
        valueSuffix={` ${metric.unit}`}
        height={310}
      />
      <Caption>
        y: {metric.axis} · x: workload × concurrency cell (`chat` = open_chat, `summ` =
        long_context_summarization; `c1`/`c8` = concurrency) · {higherIsWorse ? "higher" : "lower"} is
        worse · client-observed, 24 measured requests per cell · on TTFT p50 the draft arm sits at{" "}
        {span(allTtftRatios, 2)}× the baseline while n-gram sits at {span(ngramTtftRatios, 2)}×, making
        it the only arm that is faster to first token, in every cell and at p95 too · source:
        `analysis.csv`, `smoke-screening-20260913-debug`
      </Caption>

      <Divider />

      <H3>Each speculative arm as a fraction of baseline output throughput</H3>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "draft repeat 1 ÷ baseline",
            data: throughputRatios1.map((r) => round(r, 3)),
            tone: "danger",
          },
          {
            name: "draft repeat 2 ÷ baseline",
            data: throughputRatios2.map((r) => round(r, 3)),
            tone: "warning",
          },
          {
            name: "ngram-n3 ÷ baseline",
            data: ngramRatios.map((r) => round(r, 3)),
            tone: "success",
          },
        ]}
        valueSuffix="×"
        horizontal
        height={320}
      />
      <Caption>
        x: output-throughput ratio (arm ÷ baseline, dimensionless; 1.00× = parity) · y: workload ×
        concurrency cell · no draft cell reaches 0.48×, while n-gram exceeds parity in{" "}
        {winningCells.length} of 6 cells, both `long_context_summarization` cells (
        {ngramRatios
          .filter((r) => r > 1)
          .map((r) => `${num(r, 3)}×`)
          .join(" and ")}
        ) · this is the `speedup_vs_baseline` column of `analysis.csv`, computed against the single
        baseline deployment.
      </Caption>
    </Stack>
  );
}

/** Acceptance rate (x) against measured output-throughput ratio (y), both repeats. */
function AcceptanceScatter() {
  const theme = useHostTheme();
  const width = 620;
  const height = 320;
  const m = { top: 14, right: 22, bottom: 46, left: 62 };
  const xMin = 0.1;
  const xMax = 0.85;
  const yMin = 0.3;
  const yMax = 1.2;
  const px = (v: number) => m.left + ((v - xMin) / (xMax - xMin)) * (width - m.left - m.right);
  const py = (v: number) =>
    height - m.bottom - ((v - yMin) / (yMax - yMin)) * (height - m.top - m.bottom);
  const xTicks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
  const yTicks = [0.3, 0.5, 0.7, 0.9, 1.1];

  return (
    <Stack gap={8}>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img">
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={m.left}
              x2={width - m.right}
              y1={py(t)}
              y2={py(t)}
              stroke={theme.stroke.tertiary}
              strokeWidth={1}
            />
            <text x={m.left - 8} y={py(t) + 4} textAnchor="end" fontSize={10} fill={theme.text.tertiary}>
              {num(t, 2)}×
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text
            key={t}
            x={px(t)}
            y={height - m.bottom + 16}
            textAnchor="middle"
            fontSize={10}
            fill={theme.text.tertiary}
          >
            {pct(t, 0)}
          </text>
        ))}
        <line
          x1={m.left}
          x2={width - m.right}
          y1={height - m.bottom}
          y2={height - m.bottom}
          stroke={theme.stroke.primary}
        />
        <line
          x1={m.left}
          x2={m.left}
          y1={m.top}
          y2={height - m.bottom}
          stroke={theme.stroke.primary}
        />
        <line
          x1={m.left}
          x2={width - m.right}
          y1={py(1)}
          y2={py(1)}
          stroke={theme.text.tertiary}
          strokeWidth={1}
          strokeDasharray="4 3"
        />
        <text x={width - m.right - 4} y={py(1) - 5} textAnchor="end" fontSize={10} fill={theme.text.tertiary}>
          parity with baseline
        </text>
        {CELLS.map((c, i) => {
          const x1 = px(c.draft1.acceptanceRate);
          const y1 = py(throughputRatios1[i]);
          const x2 = px(c.draft2.acceptanceRate);
          const y2 = py(throughputRatios2[i]);
          const xn = px(c.ngram.acceptanceRate);
          const yn = py(ngramRatios[i]);
          return (
            <g key={c.key}>
              <line
                x1={x1}
                x2={x2}
                y1={y1}
                y2={y2}
                stroke={theme.stroke.secondary}
                strokeWidth={1}
              />
              <circle cx={x1} cy={y1} r={5} fill={theme.accent.primary} />
              <circle
                cx={x2}
                cy={y2}
                r={5}
                fill={theme.bg.editor}
                stroke={theme.accent.primary}
                strokeWidth={1.5}
              />
              <rect
                x={xn - 4.5}
                y={yn - 4.5}
                width={9}
                height={9}
                fill={theme.text.secondary}
                transform={`rotate(45 ${xn} ${yn})`}
              />
              <text x={x1 + c.labelDx} y={y1 + c.labelDy} fontSize={10} fill={theme.text.secondary}>
                {c.short}
              </text>
              <text x={xn + 9} y={yn + 4} fontSize={10} fill={theme.text.tertiary}>
                {c.short}
              </text>
            </g>
          );
        })}
        <text
          x={m.left + (width - m.left - m.right) / 2}
          y={height - 8}
          textAnchor="middle"
          fontSize={11}
          fill={theme.text.secondary}
        >
          Acceptance rate (accepted ÷ proposed draft tokens)
        </text>
        <text
          x={14}
          y={m.top + (height - m.top - m.bottom) / 2}
          fontSize={11}
          fill={theme.text.secondary}
          transform={`rotate(-90 14 ${m.top + (height - m.top - m.bottom) / 2})`}
          textAnchor="middle"
        >
          Output throughput ratio (draft ÷ baseline)
        </text>
      </svg>
      <Row gap={14} wrap>
        <Text size="small" tone="tertiary">
          filled circle = draft repeat 1 (job 3391426)
        </Text>
        <Text size="small" tone="tertiary">
          hollow circle = draft repeat 2 (job 3392153)
        </Text>
        <Text size="small" tone="tertiary">
          diamond = ngram-n3 (job 3392370)
        </Text>
        <Text size="small" tone="tertiary">
          line joins the same cell across the two draft deployments
        </Text>
      </Row>
      <Caption>
        All 18 speculative cells. The relationship across arms is negative: the draft model accepts{" "}
        {span(allAcceptances, 3)} and returns {span(allThroughputRatios, 3)}×, while n-gram accepts{" "}
        {span(ngramAcceptances, 3)} and returns {span(ngramRatios, 3)}×. Acceptance rate therefore does
        not predict throughput across methods, which is H1's "acceptance alone is insufficient" case
        stated as strongly as this run can state it. Within the draft arm the joining lines are
        near-vertical: the second deployment lost throughput at essentially unchanged acceptance.
        Source: per-cell `summary.json` counter deltas.
      </Caption>
    </Stack>
  );
}

function AcceptanceSection() {
  return (
    <Stack gap={12}>
      <H2>Acceptance does not buy throughput, and across methods it inverts</H2>
      <Text tone="secondary">
        The 8B drafter has {span(allMals)} tokens accepted per verification step out of a possible 4
        (3 drafted + 1 bonus); n-gram manages only {span(CELLS.map((c) => c.ngram.meanAcceptanceLength))}.
        If a speculative step cost the same as a baseline decode step, those numbers would be the
        speedups. Neither arm attains its bound, and the arm further from its bound is the one that
        wins: the drafter's step is far more expensive, which the{" "}
        <Text italic>step-cost decomposition below</Text> quantifies. The measured stream-event gap of
        ~91–101 ms in the draft deployments, against ~17–20 ms for n-gram and ~14 ms in the baseline,
        is the same quantity observed directly at the client.
      </Text>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "Upper bound from mean acceptance length, draft repeat 1 (derived)",
            data: CELLS.map((c) => round(c.draft1.meanAcceptanceLength, 2)),
            tone: "neutral",
          },
          {
            name: "Measured output-throughput ratio, draft repeat 1",
            data: throughputRatios1.map((r) => round(r, 3)),
            tone: "danger",
          },
          {
            name: "Upper bound from mean acceptance length, ngram-n3 (derived)",
            data: CELLS.map((c) => round(c.ngram.meanAcceptanceLength, 2)),
            tone: "info",
          },
          {
            name: "Measured output-throughput ratio, ngram-n3",
            data: ngramRatios.map((r) => round(r, 3)),
            tone: "success",
          },
        ]}
        valueSuffix="×"
        height={300}
      />
      <Caption>
        y: throughput relative to baseline (×, dimensionless) · x: workload × concurrency cell · the
        two bound series are{" "}
        <Text size="small" italic tone="tertiary">
          derived
        </Text>
        , not measured: each is that arm's mean acceptance length, i.e. the speedup that would follow
        only if its speculative step were as cheap as one baseline decode step. The draft arm has the
        higher bound and by far the larger shortfall; n-gram's bound is low but nearly half of it
        survives.
      </Caption>

      <Divider />

      <Grid columns="1.15fr 1fr" gap={20} align="start">
        <AcceptanceScatter />
        <Stack gap={8}>
          <H3>Mean acceptance length per cell</H3>
          <BarChart
            categories={CATEGORIES}
            series={[
              {
                name: "draft repeat 1",
                data: CELLS.map((c) => round(c.draft1.meanAcceptanceLength, 3)),
                tone: "danger",
              },
              {
                name: "draft repeat 2",
                data: CELLS.map((c) => round(c.draft2.meanAcceptanceLength, 3)),
                tone: "warning",
              },
              {
                name: "ngram-n3",
                data: CELLS.map((c) => round(c.ngram.meanAcceptanceLength, 3)),
                tone: "success",
              },
            ]}
            valueSuffix=" tok"
            height={264}
          />
          <Caption>
            y: tokens accepted per verification step, `1 + accepted/drafts`, including the bonus token
            (maximum 4 at `num_speculative_tokens=3`) · x: workload × concurrency cell · for the
            drafter `code` is highest and the two deployments sit on top of each other, the stability
            H2 predicts in direction; for n-gram the ranking is different — summarization leads and
            `code` is no better than open chat, because prompt lookup rewards outputs that quote the
            prompt, not prompts about code.
          </Caption>
        </Stack>
      </Grid>

      <Divider />

      <H3>Per-position acceptance, draft model against n-gram</H3>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "draft repeat 1, position 0",
            data: CELLS.map((c) => round(c.draft1.perPosition[0] * 100, 1)),
          },
          {
            name: "draft repeat 1, position 1",
            data: CELLS.map((c) => round(c.draft1.perPosition[1] * 100, 1)),
          },
          {
            name: "draft repeat 1, position 2",
            data: CELLS.map((c) => round(c.draft1.perPosition[2] * 100, 1)),
          },
          {
            name: "ngram-n3, position 0",
            data: CELLS.map((c) => round(c.ngram.perPosition[0] * 100, 1)),
          },
          {
            name: "ngram-n3, position 1",
            data: CELLS.map((c) => round(c.ngram.perPosition[1] * 100, 1)),
          },
          {
            name: "ngram-n3, position 2",
            data: CELLS.map((c) => round(c.ngram.perPosition[2] * 100, 1)),
          },
        ]}
        valueSuffix="%"
        height={320}
      />
      <Caption>
        y: acceptance (% of drafted tokens at that position) · x: workload × concurrency cell ·
        positions are vLLM's 0/1/2 within each 3-token proposal · draft repeat 2 is omitted here and
        tabulated below · acceptance decays with position in every cell of every arm, from 88.6→70.3%
        for the drafter on `code` at c1 down to 21.6→6.6% for n-gram on the same cell · source:
        `vllm:spec_decode_num_accepted_tokens_per_pos_total` deltas in each cell's `summary.json`.
      </Caption>

      <Card>
        <CardHeader trailing="all three speculative deployments">
          speculative-decoding counter deltas
        </CardHeader>
        <CardBody style={{ padding: 0 }}>
          <Table
            framed={false}
            headers={[
              "Workload",
              "Conc.",
              "Deployment",
              "Drafts",
              "Draft tokens",
              "Accepted",
              "Acceptance",
              "Mean acc. length",
              "Per-position (0/1/2)",
            ]}
            columnAlign={[
              "left",
              "right",
              "left",
              "right",
              "right",
              "right",
              "right",
              "right",
              "right",
            ]}
            rowTone={CELLS.flatMap(() => [undefined, undefined, "success" as const])}
            rows={CELLS.flatMap((c) =>
              (
                [
                  ["draft r1", c.draft1],
                  ["draft r2", c.draft2],
                  ["ngram", c.ngram],
                ] as Array<[string, DraftArm]>
              ).map(([label, arm]) => [
                c.workload,
                c.concurrency,
                label,
                arm.drafts.toLocaleString("en-US"),
                arm.draftTokens.toLocaleString("en-US"),
                arm.acceptedTokens.toLocaleString("en-US"),
                num(arm.acceptanceRate, 3),
                num(arm.meanAcceptanceLength, 2),
                arm.perPosition.map((p) => num(p, 3)).join(" / "),
              ]),
            )}
            striped
          />
        </CardBody>
      </Card>
      <Caption>
        Deltas across each measured window, from `/metrics` snapshots taken immediately before and
        after the cell. No counter decreased in any window, so no window was rejected for a server
        restart.
      </Caption>
    </Stack>
  );
}

function StepCost() {
  return (
    <Stack gap={12}>
      <H2>Where the speculative step cost goes</H2>
      <Text tone="secondary">
        Speculation changes how many tokens leave the engine per step, so per-token latency
        understates per-step cost. With one arm whose proposals cost no forward pass, the cost splits.
        Mean step time is recovered from the counters as `mean TPOT × T ÷ (T − accepted)`, where `T` is
        the cell's completion tokens: total steps are `T − accepted`, because every accepted draft
        token is a token that needed no step of its own. The baseline emits one token per step, so its
        TPOT is its step time. All of this is <Text italic>derived</Text> from committed per-cell
        counters and mean TPOT, not measured directly.
      </Text>
      <Grid columns="1.35fr 1fr" gap={20} align="start">
        <Stack gap={8}>
          <BarChart
            categories={CATEGORIES}
            series={[
              {
                name: "baseline decode step",
                data: baselineSteps.map((v) => round(v, 1)),
                tone: "info",
              },
              {
                name: "ngram-n3 step (verify only)",
                data: ngramSteps.map((v) => round(v, 1)),
                tone: "success",
              },
              {
                name: "draft-n3-tp4 step, repeat 1",
                data: draft1Steps.map((v) => round(v, 1)),
                tone: "danger",
              },
              {
                name: "draft-n3-tp4 step, repeat 2",
                data: draft2Steps.map((v) => round(v, 1)),
                tone: "warning",
              },
            ]}
            valueSuffix=" ms"
            height={300}
          />
          <Caption>
            y: mean engine step time (ms), derived · x: workload × concurrency cell · the n-gram bar
            is one target forward pass plus verification of three proposed tokens; the draft bars add
            the 8B drafter on top of that.
          </Caption>
        </Stack>
        <Stack gap={12}>
          <Grid columns={2} gap={16}>
            <Stat
              value={`${span(verifyCost, 2)}×`}
              label="verification + scheduling change, in baseline decode steps"
              tone="success"
            />
            <Stat
              value={`${span(draftCost, 1)}×`}
              label="the 8B drafter's additional cost, in baseline decode steps"
              tone="danger"
            />
          </Grid>
          <Text tone="secondary">
            Verifying a depth-3 proposal costs {span(verifyCost, 2)}× a plain decode step. Proposing
            with the 8B drafter costs a further {span(draftCost, 1)}×, taking the whole step to{" "}
            {span(draftStepCost, 1)}×. Draft cost dominates verification by an order of magnitude,
            which locates the regression — and it is also about an order of magnitude more than three
            8B forward passes at TP=4 can account for, so it is draft-path overhead (separate model
            execution, launch and synchronisation, drafter steps that appear not to use the target's
            captured graphs) rather than drafter arithmetic or tensor-parallel collectives.
          </Text>
        </Stack>
      </Grid>
      <Table
        headers={[
          "Workload",
          "Conc.",
          "Baseline step",
          "N-gram step",
          "Draft step r1",
          "Draft step r2",
          "Verify cost",
          "Draft cost added r1 / r2",
          "Draft step total r1 / r2",
        ]}
        columnAlign={[
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
        ]}
        rows={CELLS.map((c, i) => [
          c.workload,
          c.concurrency,
          `${num(baselineSteps[i], 2)} ms`,
          `${num(ngramSteps[i], 2)} ms`,
          `${num(draft1Steps[i], 2)} ms`,
          `${num(draft2Steps[i], 2)} ms`,
          `${num(verifyCost[i], 2)}×`,
          `${num(draftCost[i], 2)}× / ${num(draftCost[i + CELLS.length], 2)}×`,
          `${num(draftStepCost[i], 2)}× / ${num(draftStepCost[i + CELLS.length], 2)}×`,
        ])}
        striped
      />
      <Callout tone="warning" title="What the decomposition assumes">
        <Text size="small">
          That mean TPOT over a cell is a fair average step cost, that verification costs the same in
          both speculative arms, and that arms are comparable across nodes. The second is imperfect in
          a known direction: `draft_model` runs with a 7168-token per-step budget while n-gram keeps
          the baseline's 8192, so part of the draft cost is budget rather than drafter. The third is
          also imperfect, though n-gram and draft repeat 2 share node nid006687, and against repeat 2
          the gap is wider, so node effects do not explain it away. Source:
          `results/smoke-screening-20260913-debug/README.md`, section "Step-cost decomposition".
        </Text>
      </Callout>
    </Stack>
  );
}

function Correctness() {
  const [compared, comparedTwo, control] = CORRECTNESS.pairs;
  return (
    <Stack gap={12}>
      <H2>The greedy correctness gate measured nondeterminism, not correctness</H2>
      <Text tone="secondary">
        With greedy decoding, speculative decoding should reproduce the target model's tokens, so the
        project's original gate required exact agreement on the six smoke prompts. It did not pass —
        and the second draft deployment showed why that failure says nothing about speculation.
      </Text>
      <Grid columns={3} gap={16}>
        <Stat
          value={`${compared.exactMatches} of ${CORRECTNESS.totalCases}`}
          label="baseline vs. draft repeat 1, exact greedy matches"
          tone="warning"
        />
        <Stat
          value={`${comparedTwo.exactMatches} of ${CORRECTNESS.totalCases}`}
          label="baseline vs. draft repeat 2, exact greedy matches"
          tone="warning"
        />
        <Stat
          value={`${control.exactMatches} of ${CORRECTNESS.totalCases}`}
          label="draft repeat 1 vs. draft repeat 2 — two identical configurations"
          tone="danger"
        />
      </Grid>
      <Callout tone="info" title="What the triple demonstrates">
        <Text size="small">
          The third number is the decisive one. Those two deployments ran the same speculative
          configuration, so they must agree if exact agreement is achievable at all — and they agree
          on nothing, as often as speculation disagrees with the baseline. That rules out a
          rejection-sampling bug and identifies the gate itself as the defect: it was measuring
          cross-deployment floating-point nondeterminism (different nodes, different batch shapes, a
          per-step token budget of 7168 against the baseline's 8192, prefix caching enabled).
          Divergent pairs share 151–564 characters of prefix and then continue differently but
          fluently, with completion-token counts within 3. Reading the 0 of 6 as "speculation is
          lossy here" is not supported by this data.
        </Text>
      </Callout>
      <H3>Re-analysis under the calibrated criterion</H3>
      <Text tone="secondary">
        The gate now measures per-prompt divergence and judges a compared pair against a{" "}
        <Text italic>calibration control</Text> — two captures of a single configuration, which
        measures exactly the nondeterminism that must not be attributed to speculation. Both
        baseline-versus-draft comparisons land inside the control's envelope. On mean normalized edit
        distance the control is the worst of those three pairs. The n-gram arm was launched once, so
        it has a divergence profile and no verdict.
      </Text>
      <Table
        headers={[
          "Pair",
          "Role",
          "Exact",
          "Common prefix, worst (chars)",
          "Prefix fraction, worst",
          "Norm. edit distance, mean",
          "Norm. edit distance, worst",
          "Token diff, worst",
        ]}
        columnAlign={["left", "left", "right", "right", "right", "right", "right", "right"]}
        rowTone={CORRECTNESS.pairs.map((p) =>
          p.role === "calibration control" ? "info" : p.verdict === "not judged" ? "warning" : undefined,
        )}
        rows={CORRECTNESS.pairs.map((p) => [
          p.label,
          p.role,
          `${p.exactMatches}/${CORRECTNESS.totalCases}`,
          p.worstPrefixCharacters,
          num(p.worstPrefixFraction, 3),
          num(p.meanEditDistance, 3),
          num(p.worstEditDistance, 3),
          p.worstTokenDifference,
        ])}
      />
      <Caption>
        Character-level statistics over the six greedy smoke captures, all of which succeeded on every
        deployment. Edit distance is Levenshtein normalized by the longer output; prefix fraction is
        the longest common prefix over the shorter output; token difference is |Δ completion_tokens|.
        The gate passes a compared pair when its worst case stays inside the control's worst case plus
        a declared margin: prefix fraction ≥ {num(CORRECTNESS.allowance.prefixFraction, 3)}, edit
        distance ≤ {num(CORRECTNESS.allowance.editDistance, 3)}, token difference ≤{" "}
        {CORRECTNESS.allowance.tokenDifference}. Both compared pairs return
        `within_control_envelope`. The n-gram row is measured but unjudged: `ngram-n3` ran once, and
        borrowing the drafter's control would flip the answer on one statistic — its worst edit
        distance, {num(CORRECTNESS.pairs[3].worstEditDistance, 3)}, sits just outside the allowance
        computed from a control of a different configuration. That is why the gate demands a
        same-configuration control rather than the nearest one available. Source:
        `results/correctness/smoke-20260913-debug/` (`comparison*.json` for the exact-match counts,
        `divergence-*.json` and `gate-*.json` for the rest).
      </Caption>
      <Grid columns={2} gap={16} align="start">
        <Callout tone="success" title="What the gate can now establish">
          <Text size="small">
            That the divergence between the baseline and a speculative deployment is not
            distinguishable from the divergence between two deployments that must agree. This run
            therefore shows{" "}
            <Text size="small" weight="semibold">
              no evidence that speculation changes the output distribution
            </Text>
            , which is consistent with losslessness.
          </Text>
        </Callout>
        <Callout tone="warning" title="What it cannot establish">
          <Text size="small">
            It is not a proof of the lossless theorem, and a pass is only as strong as the
            nondeterminism the control exercised.{" "}
            <Text size="small" weight="semibold">
              A same-deployment paired check is impossible by construction here
            </Text>
            : `speculative_config` is fixed at launch, so one deployment serves
            exactly one configuration and can never host both arms. Even a same-node pair keeps a
            different per-step token budget and async-scheduling setting, so batch shapes differ
            because speculation is enabled. The control here is a speculative pair, so baseline-side
            variation is unmeasured; a baseline-vs-baseline control needs a second baseline
            deployment, which this run does not have.
          </Text>
        </Callout>
      </Grid>
    </Stack>
  );
}

function DraftTpOne() {
  return (
    <Stack gap={12}>
      <H2>`draft_tensor_parallel_size=1` cannot be served on this stack</H2>
      <Grid columns="1.25fr 1fr" gap={20} align="start">
        <Stack gap={10}>
          <Text tone="secondary">
            The next planned cell after the depth-3 TP=4 pair was the same configuration with the
            drafter unsharded, to test whether tensor-parallel communication on every draft step
            explains the regression. Job 3392110 reached vLLM with the flag intact and all four
            workers raised at engine construction, about 2.5 minutes into the allocation:
          </Text>
          <Card>
            <CardBody>
              <Text size="small" style={{ fontFamily: "monospace", lineHeight: 1.5 }}>
                ValueError: Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' must be
                the same. Got 1 and 4. Please pass 'draft_tensor_parallel_size' in the
                speculative_config.
              </Text>
            </CardBody>
          </Card>
          <Text tone="secondary">
            The guard is `_raise_if_draft_tp_mismatch` at `vllm/v1/spec_decode/draft_model.py:74`,
            called unconditionally from `DraftModelProposer.__init__`. Its own comment gives the
            reason: with target TP&gt;1 and draft TP=1 every rank compiles the draft model as rank 0,
            which overwrites and corrupts the `torch.compile` cache. Mismatched draft TP is therefore
            not implemented at the pinned revision, and deleting the guard with another bind-mount
            would produce a run of unknown validity rather than a measurement.
          </Text>
        </Stack>
        <Stack gap={12}>
          <Callout tone="danger" title="H4 and the draft-TP half of RQ3 are unanswerable">
            <Text size="small">
              H4 predicted TP=1 trades draft communication for memory/compute imbalance. The level does
              not exist at this revision, so research question 3 reduces to a depth sweep at draft TP =
              target TP. `configs/experiment.yaml` now pins `draft_tensor_parallel_size: [4]`.
            </Text>
          </Callout>
          <Callout tone="success" title="The n-gram arm answered the question this cell was asked">
            <Text size="small">
              The cell existed to test whether draft-side work drives the regression. The n-gram arm
              tested it instead, by removing the draft forward pass while keeping verification: draft
              cost is {span(draftCost, 1)}× a decode step against {span(verifyCost, 2)}× for
              verification. The cost is real and it is in the draft path — but at that magnitude it
              cannot be tensor-parallel collectives, so eliminating draft communication could have
              recovered only a slice of it. Draft TP=1 is a lost factor level, no longer a missing
              explanation.
            </Text>
          </Callout>
          <Caption>
            The failure precedes KV-cache profiling, so no context length was changed and the
            KV-capacity question at draft TP=1 is still open. Evidence:
            `results/deployment-failures/draft-n3-tp1-3392110/` (job 3392110, nid006593, with
            `engine-error.log` and `replica_health.json`). A failure that removes a factor level from
            the design is a result.
          </Caption>
        </Stack>
      </Grid>
    </Stack>
  );
}

function MemorySection() {
  const kvRatio = KV_CACHE.draft1.tokens / KV_CACHE.baseline.tokens;
  return (
    <Stack gap={12}>
      <H2>What speculation costs in KV cache, and what the drafter costs on top</H2>
      <Grid columns="1fr 1fr" gap={20} align="start">
        <Stack gap={8}>
          <BarChart
            categories={["baseline", "ngram-n3", "draft repeat 1", "draft repeat 2"]}
            series={[
              {
                name: "Advertised KV-cache capacity",
                data: [
                  KV_CACHE.baseline.tokens,
                  KV_CACHE.ngram.tokens,
                  KV_CACHE.draft1.tokens,
                  KV_CACHE.draft2.tokens,
                ],
              },
            ]}
            valueSuffix=" tok"
            height={230}
          />
          <Caption>
            y: KV-cache capacity (tokens) · x: deployment · all four ran at
            `gpu_memory_utilization=0.8` on 4 GH200 · source: `kv_cache_size_tokens` in
            `vllm:cache_config_info`, `metrics_after.prom`.
          </Caption>
        </Stack>
        <Stack gap={12}>
          <Grid columns={2} gap={16}>
            <Stat
              value={(KV_CACHE.baseline.tokens - KV_CACHE.ngram.tokens).toLocaleString("en-US")}
              label="tokens given up by enabling speculation, no drafter resident"
              tone="warning"
            />
            <Stat
              value={(KV_CACHE.ngram.tokens - KV_CACHE.draft1.tokens).toLocaleString("en-US")}
              label="further tokens given up to the drafter's weights"
              tone="danger"
            />
            <Stat
              value={`${num(KV_CACHE.ngram.maxConcurrency)}×`}
              label="ngram-n3 max concurrency at 131072 tok/request"
              tone="warning"
            />
            <Stat
              value={`${num(KV_CACHE.draft1.maxConcurrency)}×`}
              label="draft-n3-tp4 max concurrency at 131072 tok/request"
              tone="danger"
            />
          </Grid>
          <Text tone="secondary">
            The memory cost splits the same way the step cost does, and not as expected. Speculation
            alone costs {pct(1 - KV_CACHE.ngram.tokens / KV_CACHE.baseline.tokens, 1)} of the
            baseline's advertised capacity with no drafter resident at all; the 8B weights then cost a
            further {pct(1 - KV_CACHE.draft1.tokens / KV_CACHE.ngram.tokens, 1)}, for{" "}
            {pct(1 - kvRatio, 1)} in total. At the configured `max_model_len` the draft deployments
            advertise capacity for fewer than two full-length requests. This is a capacity result
            independent of any latency measurement, and it holds even if the latency regression turns
            out to be a configuration artefact. The two draft deployments sized their cache within{" "}
            {KV_CACHE.draft2.tokens - KV_CACHE.draft1.tokens} tokens of each other, so unlike the
            throughput ratio, the memory cost reproduces exactly.
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}

const CAVEATS: Array<{ title: string; tone: "warning" | "danger"; body: Node }> = [
  {
    title: "The draft arm did not run the same image as the others",
    tone: "danger",
    body: (
      <Text size="small">
        Both draft deployments ran with `patches/vllm-apertus-image-token.patch` bind-mounted over
        `/workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py` in the pinned image, because that
        image reads `image_token_index` off the target config and Apertus 1.5 only defines
        `image_token_id`, so the drafter cannot load without it. The baseline and n-gram arms used the
        stock image, since neither loads a drafter. The fix is open upstream as swiss-ai/vllm#20 and
        unmerged.
      </Text>
    ),
  },
  {
    title: "Two serving properties are confounded with method",
    tone: "danger",
    body: (
      <Text size="small">
        vLLM auto-disables async scheduling for both speculative methods (the baseline logs
        `Asynchronous scheduling is enabled`, both speculative arms log disabled), so that effect
        separates speculation from the baseline but is common to the speculative arms. It also shrinks
        the per-step token budget to `max_num_scheduled_tokens=7168` for `draft_model` only, against
        `max_num_batched_tokens=8192` on the baseline and the n-gram arm — that one is asymmetric
        between the speculative arms and inflates any draft-versus-n-gram comparison. This is a
        measurement of the shipped configurations, not of speculation in isolation.
      </Text>
    ),
  },
  {
    title: "Only the draft arm has two deployments",
    tone: "warning",
    body: (
      <Text size="small">
        Screening asks for two independent deployments per arm. The draft arm has them; the baseline
        and n-gram arms have one each, so they contribute no deployment-level spread and every ratio
        here inherits the uncertainty of a single baseline. Two repeats also support a spread but not
        the bootstrap confidence interval over deployment effects that the protocol's analysis section
        asks for — and the n-gram result, the most interesting one, rests on a single deployment.
      </Text>
    ),
  },
  {
    title: "Different nodes, different times",
    tone: "warning",
    body: (
      <Text size="small">
        Baseline job 3391425 ran on nid007645, draft repeat 1 (3391426) on nid006633, draft repeat 2
        (3392153) and n-gram (3392370) on nid006687, sequentially, because the `debug` QOS permits one
        running job at a time (the second queued with reason `QOSMaxJobs`). Node and time effects are
        not separated from `method`, and repeat 2 being uniformly slower is consistent with a node
        effect. The n-gram arm sharing a node with draft repeat 2 is the one node-matched pairing here.
      </Text>
    ),
  },
  {
    title: "The prefill-heavy regime is untested",
    tone: "warning",
    body: (
      <Text size="small">
        The smoke corpus' "long context" stratum records 6,780 prompt tokens across the 24 requests of
        a cell, about 283 tokens per request — far below the 16k–64k documents the research corpus
        specifies. These `long_context_summarization` rows therefore measure almost no prefill, which
        is the regime where speculation has the most room to help TTFT.
      </Text>
    ),
  },
  {
    title: "One cell breaches the repeat-spread gate",
    tone: "warning",
    body: (
      <Text size="small">
        `long_context_summarization` at concurrency 1 moves{" "}
        {pct(ratioSpreads[CELLS.findIndex((c) => c.key === "summ-c1")], 1)} between the two draft
        deployments, against `maximum_repeat_spread_fraction: 0.10`. The regression's direction is
        unaffected — the draft arm returns {span(allThroughputRatios, 3)}× baseline output throughput
        across all twelve cells, far outside deployment-level variability — but that cell's effect size
        is the least settled of the six, and it is also the stratum where n-gram wins, so the
        comparison that matters most is the one with the loosest spread.
      </Text>
    ),
  },
];

function Caveats() {
  return (
    <Stack gap={12}>
      <H2>What this run cannot claim</H2>
      <Text tone="secondary">
        These limits are load-bearing, not footnotes: two of them push in the same direction as the
        measured regression, and one removes a factor level from the design entirely.
      </Text>
      <Grid columns={2} gap={16} align="start">
        {CAVEATS.map((c) => (
          <Callout key={c.title} tone={c.tone} title={c.title}>
            {c.body}
          </Callout>
        ))}
      </Grid>
      <Card>
        <CardHeader>not measured in this run</CardHeader>
        <CardBody>
          <Grid columns={2} gap={8}>
            {[
              "concurrency 32 and the confirmation matrix",
              "speculative depths other than 3, in either method",
              "draft TP=1 — unservable at the pinned vLLM revision",
              "an `ignore_eos` control, so output lengths are not fixed",
              "the open-loop serving-capacity / sustainable-rate search",
              "GPU utilization, power, SM activity, communication metrics",
              "a second baseline or n-gram deployment",
              "a correctness gate on the n-gram arm, which has no control yet",
              "gateway latency — the client hit the replica node IP directly",
            ].map((t) => (
              <Text key={t} size="small" tone="secondary">
                {t}
              </Text>
            ))}
          </Grid>
        </CardBody>
      </Card>
      <Caption>
        Request-level percentiles in the tables are reported as recorded, but the 24 requests per cell
        cycle through only 2 prompts per stratum, so those distributions are narrow by construction and
        the samples are not independent.
      </Caption>
    </Stack>
  );
}

function AllCells() {
  const rows = CELLS.flatMap((c) => [
    ["baseline", c, c.baseline] as const,
    ["draft-n3-tp4 r1", c, c.draft1] as const,
    ["draft-n3-tp4 r2", c, c.draft2] as const,
    ["ngram-n3", c, c.ngram] as const,
  ]).map(([arm, cell, a]) => [
    `${cell.workload} c${cell.concurrency}`,
    arm,
    "1.000",
    a.promptTokens.toLocaleString("en-US"),
    a.completionTokens.toLocaleString("en-US"),
    num(a.ttftP50, 1),
    num(a.ttftP95, 1),
    num(a.tpotP50, 2),
    num(a.tpotP95, 2),
    num(a.outTps, 1),
    num(a.e2eP50 / 1000, 2),
    num(a.gapP50, 1),
  ]);

  return (
    <Stack gap={10}>
      <H2>All 24 cells as recorded</H2>
      <Table
        headers={[
          "Cell",
          "Deployment",
          "Success",
          "Prompt tok",
          "Compl. tok",
          "TTFT p50 (ms)",
          "TTFT p95 (ms)",
          "TPOT p50 (ms)",
          "TPOT p95 (ms)",
          "Output tok/s",
          "E2E p50 (s)",
          "Stream gap p50 (ms)",
        ]}
        columnAlign={[
          "left",
          "left",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
          "right",
        ]}
        rowTone={rows.map((r) =>
          r[1] === "draft-n3-tp4 r1"
            ? "danger"
            : r[1] === "draft-n3-tp4 r2"
              ? "warning"
              : r[1] === "ngram-n3"
                ? "success"
                : undefined,
        )}
        rows={rows}
        striped
      />
      <Caption>
        TTFT is request start to the first non-empty streamed payload; TPOT is `(E2E −
        TTFT)/(completion_tokens − 1)` per request; output throughput is completion tokens over
        whole-cell wall time. Stream gap is the interval between non-empty SSE payloads and is a
        diagnostic, not token-level latency — speculation delivers several accepted tokens in one
        event, which is why the draft deployments' ~91–101 ms gap sits next to a ~30–37 ms TPOT while
        the baseline's gap and TPOT coincide at ~14 ms. Red rows are draft repeat 1, amber draft repeat
        2, green n-gram.
      </Caption>
    </Stack>
  );
}

function Provenance() {
  const lines: Array<[string, string]> = [
    ["run directory", "results/smoke-screening-20260913-debug"],
    [
      "correctness artifacts",
      "results/correctness/smoke-20260913-debug (captures, exact-match comparisons, calibrated gate)",
    ],
    ["deployment failure", "results/deployment-failures/draft-n3-tp1-3392110"],
    ["baseline", "job 3391425, nid007645, replica head 172.28.51.237, stock image"],
    ["draft repeat 1", "job 3391426, nid006633, replica head 172.28.32.244, patched overlay"],
    ["draft repeat 2", "job 3392153, nid006687, replica head 172.28.33.172, patched overlay"],
    ["ngram-n3", "job 3392370, nid006687, stock image, prompt lookup 1–4"],
    [
      "measurement windows",
      "2026-09-13T15:35:52Z → 16:35:24Z (baseline and draft repeat 1), 17:19Z → 17:35Z (draft repeat 2), 18:00Z → 18:15Z (n-gram)",
    ],
    ["vLLM", "a601a9d998ddeb488f0c17e8512874b116aa7658, build 0.23.1rc1.dev1029+ga601a9d99"],
    [
      "container image",
      "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh",
    ],
    ["model-launch", "909026a990454557f1b54d26f24ec3ad92e51e35"],
    ["harness", "apertus-bench 0.1.0 at 270413f22f41ad008a72fffae5359cb3bd03785f, Python 3.13.9"],
    [
      "corpus",
      "workloads/smoke.jsonl, sha256 316fb566a18e32305b98929e34fdac515b9a43c6c7329994b319870f879952af",
    ],
    ["sampling", "temperature 0.0, top_p 1.0, seed 1 + request_index, natural EOS"],
    [
      "KV cache",
      "513,696 tokens baseline; 416,640 n-gram; 251,472 (draft repeat 1) and 251,488 (draft repeat 2)",
    ],
  ];
  return (
    <Card collapsible defaultOpen={false}>
      <CardHeader trailing="pins, jobs, digests">provenance</CardHeader>
      <CardBody>
        <Stack gap={6}>
          {lines.map(([k, v]) => (
            <Row key={k} gap={12} align="start">
              <Text size="small" tone="tertiary" style={{ minWidth: 160 }}>
                {k}
              </Text>
              <Text size="small" style={{ wordBreak: "break-all" }}>
                {v}
              </Text>
            </Row>
          ))}
        </Stack>
      </CardBody>
    </Card>
  );
}

export default function ApertusSpeculativeDecodingSmokeScreening() {
  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1180 }}>
      <Header />
      <Headline />
      <Divider />
      <DeploymentSpread />
      <Divider />
      <PairedComparison />
      <Divider />
      <AcceptanceSection />
      <Divider />
      <StepCost />
      <Divider />
      <Correctness />
      <Divider />
      <DraftTpOne />
      <Divider />
      <MemorySection />
      <Divider />
      <Caveats />
      <Divider />
      <AllCells />
      <Provenance />
    </Stack>
  );
}
