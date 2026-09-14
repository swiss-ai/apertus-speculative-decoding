/**
 * Paired measurement of speculative decoding for Apertus-v1.5-70B: one baseline
 * deployment against two independently launched 8B draft deployments and two
 * independently launched model-free n-gram deployments.
 *
 * Every number rendered here is read from the committed run artifacts:
 *   - `results/smoke-screening-20260913-debug/analysis.csv` and each cell's
 *     `summary.json` (client-observed metrics, speculative counter deltas),
 *   - `metrics_after.prom` (`vllm:cache_config_info`) for KV-cache capacity,
 *   - `results/correctness/smoke-20260913-debug/` (exact-match comparisons and
 *     the calibrated divergence gate that replaced them),
 *   - `results/deployment-failures/draft-n3-tp1-3392110/` (unservable draft TP).
 * Derived quantities are labelled as derived. Nothing is extrapolated.
 *
 * The configuration diff in `ConfigDiff` is read from
 * `examples/clariden/cli/swiss-ai/apertus-ai-1.5-release/Apertus-v1.5-70B-spec-decode.sh`
 * in `swiss-ai/model-launch` against this repository's `launch/draft-model.sh`.
 * Pull-request states are as of 2026-09-14. N-gram deployment repeat 2 (job
 * 3403354) landed as commit `246e069` and is included throughout.
 *
 * Narrative companion: `docs/hackathon-20260915.md`.
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
  /** `ngram-n3` (prompt lookup 1–4) deployment repeat 1, job 3392370 on nid006687. */
  ngram: DraftArm;
  /** `ngram-n3` deployment repeat 2, job 3403354 on nid007500. */
  ngram2: DraftArm;
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
    ngram2: {
      ttftP50: 25.34669346641749,
      ttftP95: 26.013556553516537,
      tpotP50: 15.694102135157863,
      tpotMean: 15.739762502905341,
      tpotP95: 16.426353003261248,
      outTps: 64.32286348214889,
      e2eP50: 2096.0855385055766,
      e2eP95: 2920.13641835656,
      gapP50: 18.377305008471012,
      promptTokens: 1944,
      completionTokens: 3240,
      drafts: 1068,
      draftTokens: 3204,
      acceptedTokens: 456,
      acceptanceRate: 0.14232209737827714,
      meanAcceptanceLength: 1.4269662921348314,
      perPosition: [0.21348314606741572, 0.1348314606741573, 0.07865168539325842],
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
    ngram2: {
      ttftP50: 38.07157545816153,
      ttftP95: 51.61533848149701,
      tpotP50: 17.021577629284337,
      tpotMean: 17.00545352950215,
      tpotP95: 17.902201187022804,
      outTps: 372.91561227000153,
      e2eP50: 2243.0368589702994,
      e2eP95: 3161.774491216056,
      gapP50: 19.226733478717506,
      promptTokens: 1944,
      completionTokens: 3240,
      drafts: 1100,
      draftTokens: 3300,
      acceptedTokens: 431,
      acceptanceRate: 0.13060606060606061,
      meanAcceptanceLength: 1.3918181818181818,
      perPosition: [0.2009090909090909, 0.12272727272727273, 0.06818181818181818],
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
    ngram2: {
      ttftP50: 24.989409488625824,
      ttftP95: 25.791620346717536,
      tpotP50: 14.480513403515573,
      tpotMean: 14.446819385740008,
      tpotP95: 15.715686004431358,
      outTps: 69.08622662889312,
      e2eP50: 5571.322993608192,
      e2eP95: 6044.344792014454,
      gapP50: 16.6152665624395,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 4032,
      draftTokens: 12048,
      acceptedTokens: 1584,
      acceptanceRate: 0.13147410358565736,
      meanAcceptanceLength: 1.3928571428571428,
      perPosition: [0.22321428571428573, 0.10416666666666667, 0.06547619047619048],
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
    ngram2: {
      ttftP50: 37.61466150172055,
      ttftP95: 53.58209606492892,
      tpotP50: 15.442115374762983,
      tpotMean: 15.499060239976805,
      tpotP95: 17.02915777366574,
      outTps: 480.2821545839667,
      e2eP50: 5951.470753410831,
      e2eP95: 6559.8203245783225,
      gapP50: 18.891832092776895,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 4102,
      draftTokens: 12252,
      acceptedTokens: 1649,
      acceptanceRate: 0.13459027097616716,
      meanAcceptanceLength: 1.401999024865919,
      perPosition: [0.22671867381764993, 0.10726474890297416, 0.06801560214529498],
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
    ngram2: {
      ttftP50: 26.275590411387384,
      ttftP95: 27.542010857723653,
      tpotP50: 11.689330598269802,
      tpotMean: 11.698615800444108,
      tpotP95: 12.384350059152235,
      outTps: 84.54541341799391,
      e2eP50: 2753.8604120491073,
      e2eP95: 3184.2010861379094,
      gapP50: 16.538404976017773,
      promptTokens: 6780,
      completionTokens: 5592,
      drafts: 2136,
      draftTokens: 6408,
      acceptedTokens: 1824,
      acceptanceRate: 0.2846441947565543,
      meanAcceptanceLength: 1.8539325842696628,
      perPosition: [0.3651685393258427, 0.2752808988764045, 0.21348314606741572],
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
    ngram2: {
      ttftP50: 39.19512196443975,
      ttftP95: 56.75843383651227,
      tpotP50: 12.854280021736162,
      tpotMean: 12.891128454177474,
      tpotP95: 13.501139006613956,
      outTps: 539.810297160561,
      e2eP50: 3038.96525898017,
      e2eP95: 3481.606187764555,
      gapP50: 19.044562941417098,
      promptTokens: 6780,
      completionTokens: 5611,
      drafts: 2172,
      draftTokens: 6516,
      acceptedTokens: 1801,
      acceptanceRate: 0.2763965623081645,
      meanAcceptanceLength: 1.8291896869244937,
      perPosition: [0.358195211786372, 0.27117863720073665, 0.1998158379373849],
    },
  },
];

/** `vllm:cache_config_info` from each deployment's `metrics_after.prom`. */
const KV_CACHE = {
  baseline: { tokens: 513696, maxConcurrency: 3.919189453125, gpuBlocks: 32106 },
  ngram: { tokens: 416640, maxConcurrency: 3.1787109375, gpuBlocks: 26040 },
  ngram2: { tokens: 416640, maxConcurrency: 3.1787109375, gpuBlocks: 26040 },
  draft1: { tokens: 251472, maxConcurrency: 1.9185791015625, gpuBlocks: 15717 },
  draft2: { tokens: 251488, maxConcurrency: 1.918701171875, gpuBlocks: 15718 },
};

/** `configs/experiment.yaml`, `validity_gates.maximum_repeat_spread_fraction`. */
const MAX_REPEAT_SPREAD = 0.1;

/**
 * `launch/draft-model.sh` against
 * `examples/clariden/cli/swiss-ai/apertus-ai-1.5-release/Apertus-v1.5-70B-spec-decode.sh`
 * in `swiss-ai/model-launch`, the starting point suggested on apertus-program#1057.
 */
const CONFIG_DIFF: Array<{
  setting: string;
  example: string;
  measured: string;
  kind: "same" | "forced" | "operational";
}> = [
  { setting: "speculative method", example: "draft_model", measured: "draft_model", kind: "same" },
  {
    setting: "draft model",
    example: "swiss-ai/Apertus-v1.5-8B",
    measured: "same, Capstor cache path",
    kind: "same",
  },
  { setting: "num_speculative_tokens", example: "3", measured: "3", kind: "same" },
  { setting: "draft_tensor_parallel_size", example: "4", measured: "4", kind: "same" },
  { setting: "tensor-parallel-size", example: "4", measured: "4", kind: "same" },
  { setting: "gpu-memory-utilization", example: "0.8", measured: "0.8", kind: "same" },
  {
    setting: "compilation-config fuse_allreduce_rms",
    example: "false",
    measured: "false",
    kind: "same",
  },
  {
    setting: "max-model-len",
    example: "262144",
    measured: "131072 — 262144 cannot reserve KV cache once the drafter is resident",
    kind: "forced",
  },
  {
    setting: "container image",
    example: "stock pinned image",
    measured: "pinned image + image_token_index overlay, or the drafter never loads",
    kind: "forced",
  },
  {
    setting: "environment toml",
    example: "packaged asset, {arch} unresolved",
    measured: "resolved copy under ~/.sml",
    kind: "forced",
  },
  {
    setting: "served-model-name",
    example: "swiss-ai/Apertus-v1.5-70B",
    measured: "unique per launch",
    kind: "operational",
  },
  {
    setting: "partition / time",
    example: "normal / 12:00:00",
    measured: "debug / 01:00:00 as run",
    kind: "operational",
  },
];

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
      key: "b-ng1",
      label: "baseline vs. n-gram repeat 1",
      role: "compared pair",
      exactMatches: 2,
      worstPrefixCharacters: 151,
      worstPrefixFraction: 0.12731871838111297,
      meanEditDistance: 0.11668092393051567,
      worstEditDistance: 0.3082865168539326,
      worstTokenDifference: 2,
      verdict: "within_control_envelope",
    },
    {
      key: "b-ng2",
      label: "baseline vs. n-gram repeat 2",
      role: "compared pair",
      exactMatches: 1,
      worstPrefixCharacters: 47,
      worstPrefixFraction: 0.04072790294627383,
      meanEditDistance: 0.21319717187838175,
      worstEditDistance: 0.3915229885057471,
      worstTokenDifference: 40,
      verdict: "exceeds_control_envelope",
    },
    {
      key: "ng1-ng2",
      label: "n-gram repeat 1 vs. repeat 2",
      role: "calibration control",
      exactMatches: 1,
      worstPrefixCharacters: 47,
      worstPrefixFraction: 0.04062229904926534,
      meanEditDistance: 0.163143940008724,
      worstEditDistance: 0.2636494252873563,
      worstTokenDifference: 38,
      verdict: "control",
    },
  ],
  totalCases: 6,
  /**
   * The `checks` allowances the gate actually applied: the tightest control's
   * worst case plus the declared margin. The draft pairs were judged against
   * the draft-repeat control alone; the n-gram pairs against both controls, and
   * the n-gram control is the tighter one on every statistic.
   */
  allowanceDraft: {
    prefixFraction: 0.07818336162988114,
    editDistance: 0.3058139534883721,
    tokenDifference: 5,
  },
  allowanceNgram: {
    prefixFraction: 0.0,
    editDistance: 0.3136494252873563,
    tokenDifference: 40,
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

const ngramArms = (cell: Cell): [DraftArm, DraftArm] => [cell.ngram, cell.ngram2];

const ngramRatios = CELLS.map((c) => ratio(c.ngram.outTps, c.baseline.outTps));
const ngramRatios2 = CELLS.map((c) => ratio(c.ngram2.outTps, c.baseline.outTps));
const allNgramRatios = [...ngramRatios, ...ngramRatios2];
const ngramAcceptances = CELLS.flatMap((c) => ngramArms(c).map((arm) => arm.acceptanceRate));
const ngramMals = CELLS.flatMap((c) => ngramArms(c).map((arm) => arm.meanAcceptanceLength));
const ngramTtftRatios = CELLS.flatMap((c) =>
  ngramArms(c).map((arm) => ratio(arm.ttftP50, c.baseline.ttftP50)),
);
const ngramTtft95Ratios = CELLS.flatMap((c) =>
  ngramArms(c).map((arm) => ratio(arm.ttftP95, c.baseline.ttftP95)),
);
const ngramOverDraft = CELLS.map((c) => ratio(c.ngram.outTps, c.draft1.outTps));
const winningCells = CELLS.filter((_, i) => ngramRatios[i] > 1 && ngramRatios2[i] > 1);
/** Cells where n-gram beat baseline on TTFT, over both deployments and both percentiles. */
const ngramTtftWins = [...ngramTtftRatios, ...ngramTtft95Ratios].filter((r) => r < 1).length;

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
const ngramSteps2 = CELLS.map((c) => stepMs(c.ngram2, c.ngram2.acceptedTokens));
const draft1Steps = CELLS.map((c) => stepMs(c.draft1, c.draft1.acceptedTokens));
const draft2Steps = CELLS.map((c) => stepMs(c.draft2, c.draft2.acceptedTokens));
/** Verification plus the scheduling change, in baseline decode steps, over both n-gram repeats. */
const verifyCost = [...ngramSteps, ...ngramSteps2].map(
  (step, i) => step / baselineSteps[i % CELLS.length],
);
/**
 * The 8B drafter's additional cost, in baseline decode steps: every draft
 * deployment against every n-gram deployment, so all four pairings per cell.
 */
const draftCost = [...draft1Steps, ...draft2Steps].flatMap((step, i) =>
  [ngramSteps[i % CELLS.length], ngramSteps2[i % CELLS.length]].map(
    (verifyStep) => (step - verifyStep) / baselineSteps[i % CELLS.length],
  ),
);
const draftCostMs = [...draft1Steps, ...draft2Steps].flatMap((step, i) =>
  [ngramSteps[i % CELLS.length], ngramSteps2[i % CELLS.length]].map(
    (verifyStep) => step - verifyStep,
  ),
);
const draftStepCost = [...draft1Steps, ...draft2Steps].map(
  (step, i) => step / baselineSteps[i % CELLS.length],
);
/** How far apart two n-gram deployments on different nodes put the same step. */
const ngramStepSpreads = CELLS.map((_, i) => spread(ngramSteps[i], ngramSteps2[i]));

const acceptanceSpreads = CELLS.map((c) => spread(c.draft1.acceptanceRate, c.draft2.acceptanceRate));
const ratioSpreads = CELLS.map((_, i) => spread(throughputRatios1[i], throughputRatios2[i]));
const breachingCells = CELLS.filter((_, i) => ratioSpreads[i] > MAX_REPEAT_SPREAD);

const ngramAcceptanceSpreads = CELLS.map((c) =>
  spread(c.ngram.acceptanceRate, c.ngram2.acceptanceRate),
);
const ngramMalSpreads = CELLS.map((c) =>
  spread(c.ngram.meanAcceptanceLength, c.ngram2.meanAcceptanceLength),
);
const ngramRatioSpreads = CELLS.map((_, i) => spread(ngramRatios[i], ngramRatios2[i]));

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
        `draft_tensor_parallel_size=4`; and model-free n-gram speculation at depth 3 with prompt
        lookup 1–4. Each speculative arm was launched twice, independently, on different nodes.
        Three workloads × two concurrencies × five deployments = 30 cells, 8 warmup then 24 measured
        requests each, all with a 1.000 success rate.
      </Text>
      <Row gap={6} wrap>
        <Pill size="sm">Clariden · partition debug</Pill>
        <Pill size="sm">2026-09-13 and 2026-09-14</Pill>
        <Pill size="sm">jobs 3391425 / 3391426 / 3392153 / 3392370 / 3403354</Pill>
        <Pill size="sm">smoke corpus, 6 prompts</Pill>
        <Pill size="sm" tone="warning">
          2 repeats per speculative arm, 1 baseline
        </Pill>
        <Pill size="sm" tone="warning">
          draft TP=1 unservable
        </Pill>
        <Pill size="sm" tone="warning">
          draft arm needs an unmerged vLLM fix
        </Pill>
      </Row>
      <Text size="small" tone="tertiary">
        Narrative companion, with reproduction commands and the group asks:
        `docs/hackathon-20260915.md`.
      </Text>
    </Stack>
  );
}

function Verdict() {
  return (
    <Grid columns="1.6fr 1fr" gap={20} align="start">
      <Callout
        tone="danger"
        title="Do not enable draft_model speculation for Apertus 1.5 70B on this stack"
      >
        <Text size="small">
          The 8B drafter accepts {spanPct(allAcceptances)} of its proposed tokens and still returns{" "}
          {span(allThroughputRatios, 3)}× baseline output throughput in all twelve of its cells,
          across two independent deployments. The cost is not acceptance and not verification: a
          speculative step with the drafter costs {span(draftStepCost, 1)}× a baseline decode step,
          of which the drafter alone is {span(draftCost, 1)}×. Three 8B forward passes at TP=4 cannot
          be five to six whole 70B steps, so that is per-step overhead in the draft path — the one
          thing worth profiling next.
        </Text>
      </Callout>
      <Callout tone="success" title="N-gram is the only positive result, and it reproduced">
        <Text size="small">
          It accepts only {spanPct(ngramAcceptances)}, and on long-context summarization it beats
          baseline in both of its independent deployments — {num(ngramRatios[4], 3)}× then{" "}
          {num(ngramRatios2[4], 3)}× at concurrency 1, {num(ngramRatios[5], 3)}× then{" "}
          {num(ngramRatios2[5], 3)}× at concurrency 8. All six cells clear the protocol's{" "}
          {pct(MAX_REPEAT_SPREAD, 0)} spread gate, worst {pct(Math.max(...ngramRatioSpreads), 1)},
          against the draft arm's {pct(Math.max(...ratioSpreads), 1)} breach. Lower TTFT than
          baseline now holds in {ngramTtftWins} of {ngramTtftRatios.length + ngramTtft95Ratios.length}{" "}
          cells across p50 and p95. What it still needs is a <Text weight="semibold">baseline</Text>{" "}
          repeat: the baseline is the denominator of every ratio here and the only arm with one
          deployment.
        </Text>
      </Callout>
    </Grid>
  );
}

function Headline() {
  return (
    <Grid columns="1.5fr 1fr" gap={20} align="start">
      <Stack gap={16}>
        <Text weight="semibold" style={{ fontSize: 17, lineHeight: 1.45 }}>
          The 8B drafter's proposed tokens are accepted at {spanPct(allAcceptances)} and it loses in all
          twelve of its cells. N-gram's acceptance rate is only {spanPct(ngramAcceptances)}, yet it is
          the only arm that wins anything, taking {span(allNgramRatios, 3)}× of baseline throughput and beating the baseline
          outright on long-context summarization in both of its deployments. Acceptance is not the
          binding constraint — the arm that accepts worst is the arm that wins.
        </Text>
        <Grid columns={3} gap={16}>
          <Stat
            value={`${span(allThroughputRatios)}×`}
            label="draft-n3-tp4 output throughput vs baseline (12 cells)"
            tone="danger"
          />
          <Stat
            value={`${span(allNgramRatios)}×`}
            label="ngram-n3 output throughput vs baseline (12 cells)"
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
        Each speculative arm was launched twice with an identical configuration, on different nodes at
        different times. The protocol's independent unit is the deployment repeat, so the gap within a
        pair is the measurement's own noise floor — and it is not the same size for every quantity,
        nor the same for the two arms. The draft arm's throughput ratio moves{" "}
        {span(ratioSpreads.map((s) => s * 100), 1)}% while its acceptance moves only{" "}
        {span(acceptanceSpreads.map((s) => s * 100), 1)}%. The n-gram arm inverts that: its throughput
        ratio moves {span(ngramRatioSpreads.map((s) => s * 100), 1)}% while its acceptance moves{" "}
        {span(ngramAcceptanceSpreads.map((s) => s * 100), 1)}%.
      </Text>
      <Grid columns="1.4fr 1fr" gap={20} align="start">
        <Stack gap={8}>
          <BarChart
            categories={CATEGORIES}
            series={[
              {
                name: "draft: acceptance rate",
                data: acceptanceSpreads.map((s) => round(s * 100, 1)),
                tone: "success",
              },
              {
                name: "draft: output-throughput ratio vs baseline",
                data: ratioSpreads.map((s) => round(s * 100, 1)),
                tone: "danger",
              },
              {
                name: "ngram: acceptance rate",
                data: ngramAcceptanceSpreads.map((s) => round(s * 100, 1)),
                tone: "warning",
              },
              {
                name: "ngram: output-throughput ratio vs baseline",
                data: ngramRatioSpreads.map((s) => round(s * 100, 1)),
                tone: "info",
              },
            ]}
            valueSuffix="%"
            height={300}
          />
          <Caption>
            y: between-deployment spread, |repeat 1 − repeat 2| ÷ mean of the two, in percent · x:
            workload × concurrency cell · lower is more reproducible · the protocol's
            `maximum_repeat_spread_fraction` is {pct(MAX_REPEAT_SPREAD, 0)}; only `{breach.workload}`
            at concurrency {breach.concurrency} exceeds it, on the draft arm's throughput ratio, at{" "}
            {pct(ratioSpreads[breachIndex], 1)} · every n-gram cell clears it, worst{" "}
            {pct(Math.max(...ngramRatioSpreads), 1)} · source: per-cell `summary.json` in all four
            speculative deployments.
          </Caption>
        </Stack>
        <Stack gap={12}>
          <Callout
            tone="success"
            title="The n-gram arm is the more reproducible of the two, including where it wins"
          >
            <Text size="small">
              Its throughput ratio spreads {span(ngramRatioSpreads.map((s) => s * 100), 1)}% against
              the draft arm's {span(ratioSpreads.map((s) => s * 100), 1)}%, and all six cells clear
              the gate. `long_context_summarization` at concurrency 1 — the cell that carries the
              headline — spreads {pct(ngramRatioSpreads[4], 1)} and wins in both deployments
              ({num(ngramRatios[4], 3)}× then {num(ngramRatios2[4], 3)}×).
            </Text>
          </Callout>
          <Callout tone="warning" title="Acceptance is a near-invariant for the drafter and not for n-gram">
            <Text size="small">
              The drafter scores fixed text with a fixed model pair, so its acceptance is a property
              of that pair: {spanPct(acceptanceSpreads, 1)} between deployments, less than its
              throughput ratio moves. Prompt lookup matches against the prompt{" "}
              <Text size="small" italic>plus the tokens generated so far</Text>, so two deployments
              whose greedy outputs diverge numerically offer different n-grams to match: n-gram's
              acceptance moves {spanPct(ngramAcceptanceSpreads, 1)}, more than its throughput ratio,
              worst in the summarization cell at concurrency 1 (
              {num(CELLS[4].ngram.acceptanceRate, 3)} → {num(CELLS[4].ngram2.acceptanceRate, 3)}).
              Mean acceptance length is steadier, {spanPct(ngramMalSpreads, 1)}.
            </Text>
          </Callout>
          <Callout tone="info" title="The between-deployment shift has no fixed direction">
            <Text size="small">
              Draft repeat 2 was slower than repeat 1 in all six of its cells; n-gram repeat 2 was
              faster than repeat 1 in all six of its own, on a fifth distinct node. So a uniform
              within-pair shift is not a "later run is slower" artefact of the harness, the partition,
              or the time of day — it behaves like a node/deployment effect, which is exactly what two
              repeats sample and cannot separate.
            </Text>
          </Callout>
        </Stack>
      </Grid>
      <Table
        headers={[
          "Workload",
          "Conc.",
          "Arm",
          "Acceptance rate r1 → r2",
          "Acceptance spread",
          "Output tok/s r1 → r2",
          "Ratio to baseline r1 → r2",
          "Ratio spread",
        ]}
        columnAlign={["left", "right", "left", "right", "right", "right", "right", "right"]}
        rowTone={CELLS.flatMap((_, i) => [
          ratioSpreads[i] > MAX_REPEAT_SPREAD ? ("danger" as const) : undefined,
          "success" as const,
        ])}
        rows={CELLS.flatMap((c, i) => [
          [
            c.workload,
            c.concurrency,
            "draft",
            `${num(c.draft1.acceptanceRate, 3)} → ${num(c.draft2.acceptanceRate, 3)}`,
            pct(acceptanceSpreads[i], 1),
            `${num(c.draft1.outTps, 1)} → ${num(c.draft2.outTps, 1)}`,
            `${num(throughputRatios1[i], 3)} → ${num(throughputRatios2[i], 3)}`,
            pct(ratioSpreads[i], 1),
          ],
          [
            c.workload,
            c.concurrency,
            "ngram",
            `${num(c.ngram.acceptanceRate, 3)} → ${num(c.ngram2.acceptanceRate, 3)}`,
            pct(ngramAcceptanceSpreads[i], 1),
            `${num(c.ngram.outTps, 1)} → ${num(c.ngram2.outTps, 1)}`,
            `${num(ngramRatios[i], 3)} → ${num(ngramRatios2[i], 3)}`,
            pct(ngramRatioSpreads[i], 1),
          ],
        ])}
        striped
      />
      <Caption>
        Every deployment ran the same corpus, request counts, warmup, sampling and load path; each
        arm's two deployments differ only in node and time. Repeat 1 of each arm was driven by
        `apertus-bench matrix`, repeat 2 by `apertus-bench run` per cell so each cell carries its
        deployment repeat index. Two repeats support a spread but not a bootstrap confidence interval
        over deployment effects, and{" "}
        <Text size="small" weight="semibold">
          the baseline still has only one deployment
        </Text>
        , so every ratio on this canvas inherits the uncertainty of a single denominator and the
        spreads above are of the numerator alone.
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
            name: "ngram-n3 repeat 1, job 3392370",
            data: CELLS.map((c) => round(c.ngram[metricId], 2)),
            tone: "success",
          },
          {
            name: "ngram-n3 repeat 2, job 3403354",
            data: CELLS.map((c) => round(c.ngram2[metricId], 2)),
            tone: "neutral",
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
        it the only arm that is faster to first token — in every cell, in both deployments, at p50 and
        p95 alike · source: `analysis.csv`, `smoke-screening-20260913-debug`
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
            name: "ngram repeat 1 ÷ baseline",
            data: ngramRatios.map((r) => round(r, 3)),
            tone: "success",
          },
          {
            name: "ngram repeat 2 ÷ baseline",
            data: ngramRatios2.map((r) => round(r, 3)),
            tone: "neutral",
          },
        ]}
        valueSuffix="×"
        horizontal
        height={360}
      />
      <Caption>
        x: output-throughput ratio (arm ÷ baseline, dimensionless; 1.00× = parity) · y: workload ×
        concurrency cell · no draft cell reaches 0.48×, while n-gram exceeds parity in{" "}
        {winningCells.length} of 6 cells <Text size="small" weight="semibold">in both deployments</Text>
        , both `long_context_summarization` cells — {num(ngramRatios[4], 3)}× then{" "}
        {num(ngramRatios2[4], 3)}× at concurrency 1, {num(ngramRatios[5], 3)}× then{" "}
        {num(ngramRatios2[5], 3)}× at concurrency 8 · this is the `speedup_vs_baseline` column of
        `analysis.csv`, computed against the single baseline deployment, so all four speculative
        series share one denominator.
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
          const xn2 = px(c.ngram2.acceptanceRate);
          const yn2 = py(ngramRatios2[i]);
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
              <line
                x1={xn}
                x2={xn2}
                y1={yn}
                y2={yn2}
                stroke={theme.stroke.secondary}
                strokeWidth={1}
              />
              <rect
                x={xn - 4.5}
                y={yn - 4.5}
                width={9}
                height={9}
                fill={theme.text.secondary}
                transform={`rotate(45 ${xn} ${yn})`}
              />
              <rect
                x={xn2 - 4.5}
                y={yn2 - 4.5}
                width={9}
                height={9}
                fill={theme.bg.editor}
                stroke={theme.text.secondary}
                strokeWidth={1.5}
                transform={`rotate(45 ${xn2} ${yn2})`}
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
          filled diamond = ngram repeat 1 (job 3392370)
        </Text>
        <Text size="small" tone="tertiary">
          hollow diamond = ngram repeat 2 (job 3403354)
        </Text>
        <Text size="small" tone="tertiary">
          line joins the same cell across one arm's two deployments
        </Text>
      </Row>
      <Caption>
        All 24 speculative cells. The relationship across arms is negative: the draft model accepts{" "}
        {span(allAcceptances, 3)} and returns {span(allThroughputRatios, 3)}×, while n-gram accepts{" "}
        {span(ngramAcceptances, 3)} and returns {span(allNgramRatios, 3)}×. Acceptance rate therefore
        does not predict throughput across methods, which is H1's "acceptance alone is insufficient"
        case stated as strongly as this run can state it. The joining lines show the two arms' noise
        structure differing: the draft pairs are near-vertical (the second deployment lost throughput
        at essentially unchanged acceptance), while the n-gram pairs tilt — acceptance and throughput
        both move, and the summarization cell at concurrency 1 moves furthest on both. Source:
        per-cell `summary.json` counter deltas.
      </Caption>
    </Stack>
  );
}

function AcceptanceSection() {
  return (
    <Stack gap={12}>
      <H2>Acceptance does not buy throughput, and across methods it inverts</H2>
      <Callout tone="info" title="Three different quantities, all called “acceptance”">
        <Stack gap={4}>
          <Text size="small">
            <Text size="small" weight="semibold">
              Acceptance rate
            </Text>{" "}
            = accepted tokens ÷ proposed draft tokens. This is `analysis.csv`'s `acceptance_rate` and
            the only quantity this canvas calls acceptance unqualified: {spanPct(allAcceptances)} for
            the drafter, {spanPct(ngramAcceptances)} for n-gram, each over two deployments.
          </Text>
          <Text size="small">
            <Text size="small" weight="semibold">
              Mean acceptance length
            </Text>{" "}
            = 1 + accepted tokens ÷ proposals, so it counts the always-accepted bonus token and is
            bounded by 1 + `num_speculative_tokens` = 4 here: {span(allMals)} for the drafter,{" "}
            {span(ngramMals)} for n-gram.
          </Text>
          <Text size="small">
            <Text size="small" weight="semibold">
              Per-position acceptance
            </Text>{" "}
            = tokens accepted at that position ÷ proposals, so the three positions sum to mean
            acceptance length minus one, and each is a share of every proposal rather than of the
            proposals that reached that position.
          </Text>
          <Text size="small" tone="tertiary">
            The first two coincide as `acceptance_rate = (mean_acceptance_length − 1) ÷ 3` only when
            every proposal is full depth. That holds for the drafter, and not quite for n-gram: on
            `code` prompt lookup sometimes proposes fewer than 3 tokens (12,444 draft tokens over
            4,164 proposals in repeat 1, 12,048 over 4,032 in repeat 2), so the two differ in the
            third decimal. That the residual is small here is a property of this corpus, not of
            n-gram, and the conversion is still the wrong one to rely on. Definitions as computed in
            `src/apertus_bench/prometheus.py` and stated in `docs/protocol.md`.
          </Text>
        </Stack>
      </Callout>
      <Text tone="secondary">
        The 8B drafter has {span(allMals)} tokens accepted per verification step out of a possible 4
        (3 drafted + 1 bonus); n-gram manages only {span(ngramMals)}. If a speculative step cost the
        same as a baseline decode step, those numbers would be the speedups. Neither arm attains its
        bound, and the arm further from its bound is the one that wins: the drafter's step is far more
        expensive, which the <Text italic>step-cost decomposition below</Text> quantifies. The
        measured stream-event gap of ~91–101 ms in the draft deployments, against ~16–20 ms across
        both n-gram deployments and ~14 ms in the baseline, is the same quantity observed directly at
        the client.
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
                name: "ngram repeat 1",
                data: CELLS.map((c) => round(c.ngram.meanAcceptanceLength, 3)),
                tone: "success",
              },
              {
                name: "ngram repeat 2",
                data: CELLS.map((c) => round(c.ngram2.meanAcceptanceLength, 3)),
                tone: "info",
              },
            ]}
            valueSuffix=" tok"
            height={264}
          />
          <Caption>
            y: mean acceptance length, `1 + accepted tokens / proposals`, including the bonus token
            (maximum 4 at `num_speculative_tokens=3`); this is not the acceptance rate · x: workload ×
            concurrency cell · for the
            drafter `code` is highest and the two deployments sit on top of each other, the stability
            H2 predicts in direction; for n-gram the ranking is different — summarization leads and
            `code` is no better than open chat, because prompt lookup rewards outputs that quote the
            prompt, not prompts about code · n-gram's two deployments track each other to{" "}
            {spanPct(ngramMalSpreads, 1)}, tighter than its acceptance rate does.
          </Caption>
        </Stack>
      </Grid>

      <Divider />

      <H3>Per-position acceptance (accepted at position ÷ proposals), draft model against n-gram</H3>
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
        y: per-position acceptance, tokens accepted at that position ÷ proposals, in percent — the
        denominator is every proposal, not the proposals that reached the position, so the three
        positions of a cell sum to its mean acceptance length minus one · x: workload × concurrency
        cell · positions are vLLM's 0/1/2 within each 3-token proposal · the two repeat-2 deployments
        are omitted here and tabulated below · acceptance decays with position in every cell of every arm, from
        88.6→70.3% for the drafter on `code` at c1 down to 21.6→6.6% for n-gram on the same cell ·
        source:
        `vllm:spec_decode_num_accepted_tokens_per_pos_total` deltas in each cell's `summary.json`.
      </Caption>

      <Card>
        <CardHeader trailing="all four speculative deployments">
          speculative-decoding counter deltas
        </CardHeader>
        <CardBody style={{ padding: 0 }}>
          <Table
            framed={false}
            headers={[
              "Workload",
              "Conc.",
              "Deployment",
              "Proposals",
              "Draft tokens",
              "Accepted",
              "Acceptance rate",
              "Mean acceptance length",
              "Per-position acceptance (0/1/2)",
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
            rowTone={CELLS.flatMap(() => [
              undefined,
              undefined,
              "success" as const,
              "success" as const,
            ])}
            rows={CELLS.flatMap((c) =>
              (
                [
                  ["draft r1", c.draft1],
                  ["draft r2", c.draft2],
                  ["ngram r1", c.ngram],
                  ["ngram r2", c.ngram2],
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
        after the cell: `vllm:spec_decode_num_drafts_total` (proposals),
        `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total` and the
        per-position counter. Acceptance rate is accepted ÷ draft tokens; mean acceptance length is
        1 + accepted ÷ proposals; per-position acceptance is accepted at that position ÷ proposals. No
        counter decreased in any window, so no window was rejected for a server restart.
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
                name: "ngram-n3 step (verify only), repeat 1",
                data: ngramSteps.map((v) => round(v, 1)),
                tone: "success",
              },
              {
                name: "ngram-n3 step (verify only), repeat 2",
                data: ngramSteps2.map((v) => round(v, 1)),
                tone: "neutral",
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
            height={320}
          />
          <Caption>
            y: mean engine step time (ms), derived · x: workload × concurrency cell · the n-gram bars
            are one target forward pass plus verification of three proposed tokens; the draft bars add
            the 8B drafter on top of that · both speculative arms now have two independent
            deployments, so every range on this slide is over two deployments rather than one
            measurement.
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
            Verifying a depth-3 proposal costs {span(verifyCost, 2)}× a plain decode step over 12
            cells from two deployments. Proposing with the 8B drafter costs a further{" "}
            {span(draftCost, 1)}× ({span(draftCostMs, 1)} ms), taking the whole step to{" "}
            {span(draftStepCost, 1)}×. Draft cost dominates verification by an order of magnitude,
            which locates the regression — and it is also about an order of magnitude more than three
            8B forward passes at TP=4 can account for, so it is draft-path overhead (separate model
            execution, launch and synchronisation, drafter steps that appear not to use the target's
            captured graphs) rather than drafter arithmetic or tensor-parallel collectives.
          </Text>
          <Callout tone="info" title="Two nodes bound the node explanation">
            <Text size="small">
              The two n-gram deployments, on nid006687 and nid007500, put step cost within{" "}
              {spanPct(ngramStepSpreads, 1)} of each other in every cell. So node-to-node variation on
              this measurement is a few percent, against a drafter cost of five to six whole baseline
              steps — a node effect cannot absorb that.
            </Text>
          </Callout>
        </Stack>
      </Grid>
      <Table
        headers={[
          "Workload",
          "Conc.",
          "Baseline step",
          "N-gram step r1 / r2",
          "N-gram step spread",
          "Draft step r1 / r2",
          "Verify cost r1 / r2",
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
        ]}
        rows={CELLS.map((c, i) => [
          c.workload,
          c.concurrency,
          `${num(baselineSteps[i], 2)} ms`,
          `${num(ngramSteps[i], 2)} / ${num(ngramSteps2[i], 2)} ms`,
          pct(ngramStepSpreads[i], 1),
          `${num(draft1Steps[i], 2)} / ${num(draft2Steps[i], 2)} ms`,
          `${num(verifyCost[i], 2)}× / ${num(verifyCost[i + CELLS.length], 2)}×`,
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
          now bounded from two directions: n-gram repeat 1 and draft repeat 2 share node nid006687 and
          the gap against draft repeat 2 is <Text size="small" italic>wider</Text> rather than
          narrower, and the two n-gram deployments agree to {spanPct(ngramStepSpreads, 1)} across
          different nodes. Source: `results/smoke-screening-20260913-debug/README.md`, section
          "Step-cost decomposition".
        </Text>
      </Callout>
    </Stack>
  );
}

function Correctness() {
  const [compared, comparedTwo, control, ngramOne, ngramTwo, ngramControl] = CORRECTNESS.pairs;
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
        baseline-versus-draft comparisons land inside the draft control's envelope. On mean normalized
        edit distance that control is the worst of those three pairs. The second n-gram deployment then
        gave the n-gram arm its own control, and the gate <Text italic>split</Text> on it.
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
          p.role === "calibration control"
            ? "info"
            : p.verdict === "exceeds_control_envelope"
              ? "danger"
              : undefined,
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
        The gate passes a compared pair when its worst case stays inside the tightest control's worst
        case plus a declared margin. For the draft pairs that allowance is a worst edit distance of{" "}
        {num(CORRECTNESS.allowanceDraft.editDistance, 3)} and a token difference of{" "}
        {CORRECTNESS.allowanceDraft.tokenDifference}, and both return `within_control_envelope`. For
        the n-gram pairs the n-gram control is tighter and so sets the allowance:{" "}
        {num(CORRECTNESS.allowanceNgram.editDistance, 3)} and{" "}
        {CORRECTNESS.allowanceNgram.tokenDifference}. Repeat 1 comes in at{" "}
        {num(ngramOne.worstEditDistance, 3)} and passes; repeat 2 comes in at{" "}
        {num(ngramTwo.worstEditDistance, 3)} and fails that one check of four. Source:
        `results/correctness/smoke-20260913-debug/` (`comparison*.json` for the exact-match counts,
        `divergence-*.json` and `gate-*.json` for the rest).
      </Caption>
      <Callout
        tone="danger"
        title="Do not present “the gate passes for n-gram”: two deployments of one configuration got opposite verdicts"
      >
        <Text size="small">
          The n-gram control ({ngramControl.exactMatches}/{CORRECTNESS.totalCases} exact, worst edit
          distance {num(ngramControl.worstEditDistance, 3)}) allows{" "}
          {num(CORRECTNESS.allowanceNgram.editDistance, 3)}. Baseline-versus-repeat-1 sits at{" "}
          {num(ngramOne.worstEditDistance, 3)} and returns `within_control_envelope`;
          baseline-versus-repeat-2 sits at {num(ngramTwo.worstEditDistance, 3)} and returns
          `exceeds_control_envelope`, passing the other three checks — with its token difference
          exactly on the allowance, {ngramTwo.worstTokenDifference} against{" "}
          {CORRECTNESS.allowanceNgram.tokenDifference}. Nothing about the configuration differs
          between those two deployments, so the split is the instrument, not the arm: the allowance is
          a worst-of-six order statistic from a single control pair, so it is itself noisy, and adding
          the draft-repeat pair as a second control does not loosen it because the gate takes the
          tightest envelope. Six prompts cannot adjudicate losslessness at this resolution.
        </Text>
      </Callout>
      <Grid columns={2} gap={16} align="start">
        <Callout tone="success" title="What the gate can establish">
          <Text size="small">
            That the divergence between the baseline and a speculative deployment is of the same
            character, and roughly the same size, as the divergence between two deployments of one
            configuration that must agree. For the draft arm both compared pairs land inside their
            control's envelope, so this run shows{" "}
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
            because speculation is enabled. Both controls are speculative pairs, so baseline-side
            variation is unmeasured; a baseline-vs-baseline control needs a second baseline
            deployment, which this run does not have — another reason the cheapest useful next launch
            is a baseline repeat.
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
            categories={[
              "baseline",
              "ngram repeat 1",
              "ngram repeat 2",
              "draft repeat 1",
              "draft repeat 2",
            ]}
            series={[
              {
                name: "Advertised KV-cache capacity",
                data: [
                  KV_CACHE.baseline.tokens,
                  KV_CACHE.ngram.tokens,
                  KV_CACHE.ngram2.tokens,
                  KV_CACHE.draft1.tokens,
                  KV_CACHE.draft2.tokens,
                ],
              },
            ]}
            valueSuffix=" tok"
            height={250}
          />
          <Caption>
            y: KV-cache capacity (tokens) · x: deployment · all five ran at
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
            The memory cost splits the same way the step cost does, and not as expected. All three
            shares below are of the <Text weight="semibold">baseline's</Text> advertised capacity, so
            they add up: speculation alone costs{" "}
            {pct(1 - KV_CACHE.ngram.tokens / KV_CACHE.baseline.tokens, 1)} with no drafter resident at
            all, the 8B weights cost a further{" "}
            {pct((KV_CACHE.ngram.tokens - KV_CACHE.draft1.tokens) / KV_CACHE.baseline.tokens, 1)}, and
            the total is {pct(1 - kvRatio, 1)}. Put the other way, of everything the draft arm gives
            up, {pct((KV_CACHE.baseline.tokens - KV_CACHE.ngram.tokens) / (KV_CACHE.baseline.tokens - KV_CACHE.draft1.tokens), 0)}{" "}
            is speculation machinery and{" "}
            {pct((KV_CACHE.ngram.tokens - KV_CACHE.draft1.tokens) / (KV_CACHE.baseline.tokens - KV_CACHE.draft1.tokens), 0)}{" "}
            is drafter weights. At the configured `max_model_len` the draft deployments
            advertise capacity for fewer than two full-length requests. This is a capacity result
            independent of any latency measurement, and it holds even if the latency regression turns
            out to be a configuration artefact.
          </Text>
          <Callout tone="success" title="The most reproducible numbers in the study">
            <Text size="small">
              The two n-gram deployments sized their cache to the same{" "}
              {KV_CACHE.ngram2.tokens.toLocaleString("en-US")} tokens on different nodes, identically,
              and the two draft deployments came within{" "}
              {KV_CACHE.draft2.tokens - KV_CACHE.draft1.tokens} tokens of each other. So unlike the
              throughput ratio, these capacity costs are deterministic properties of the
              configuration rather than sizing accidents.
            </Text>
          </Callout>
        </Stack>
      </Grid>
    </Stack>
  );
}

function ConfigDiff() {
  const forced = CONFIG_DIFF.filter((r) => r.kind === "forced");
  const same = CONFIG_DIFF.filter((r) => r.kind === "same");
  return (
    <Stack gap={12}>
      <H2>Against the published `Apertus-v1.5-70B-spec-decode.sh`</H2>
      <Text tone="secondary">
        The starting point suggested on apertus-program#1057 is `model-launch`'s own
        speculative-decoding example. Its {same.length} speculative and parallelism settings are
        reproduced exactly, so the numbers on this canvas are numbers for the published
        configuration. {forced.length} settings had to change, and each one is a bug or a limit
        rather than a tuning choice — which means{" "}
        <Text weight="semibold">the example as published does not currently run.</Text>
      </Text>
      <Table
        headers={["Setting", "Published example", "As measured here", "Why"]}
        columnAlign={["left", "left", "left", "left"]}
        rowTone={CONFIG_DIFF.map((r) => (r.kind === "forced" ? "danger" : undefined))}
        rows={CONFIG_DIFF.map((r) => [
          r.setting,
          r.example,
          r.measured,
          r.kind === "same" ? "identical" : r.kind === "forced" ? "forced" : "operational",
        ])}
        striped
      />
      <Callout tone="warning" title="The example's header comment claims the outputs are lossless">
        <Text size="small">
          “Lossless — outputs match the plain 70B” is a theorem about exact arithmetic. On this stack
          two identically configured draft deployments agree on{" "}
          {CORRECTNESS.pairs[2].exactMatches} of {CORRECTNESS.totalCases} greedy prompts, so “outputs
          match” is not a checkable property of a launch here and should not be advertised as one.
          The supportable statement is the calibrated one below: divergence from the baseline is not
          distinguishable from divergence between two deployments that must agree.
        </Text>
      </Callout>
    </Stack>
  );
}

function PlatformFindings() {
  return (
    <Stack gap={12}>
      <H2>Platform findings that affect anyone else trying this</H2>
      <Grid columns={2} gap={16} align="start">
        <Callout
          tone="danger"
          title="draft_model with an Apertus 1.5 target crashes at drafter load on the pinned image"
        >
          <Text size="small">
            vLLM's `SpecDecodeBaseProposer.load_model()` reads `image_token_index` off the target
            config for any architecture not on an allow-list.
            `Apertus1p5ForConditionalGeneration` is multimodal but absent from that list, and
            `Apertus1p5Config` defines `image_token_id` instead — so the 70B loads, the drafter
            raises `AttributeError`, and the engine never comes up. The one-line fix is
            `patches/vllm-apertus-image-token.patch`, bind-mounted over the read-only image by
            `launch/patch-vllm.sh`, and is open as swiss-ai/vllm#20 against `apertus-1-5`.
          </Text>
        </Callout>
        <Callout tone="danger" title="No image contains the fix, and the upstreaming PRs do not carry it">
          <Text size="small">
            swiss-ai/vllm#20 is unmerged, and the maintainers' position is that `apertus-1-5` is the
            reference branch for the upstream PR and should not take experimental changes. Neither
            swiss-ai/vllm#16 nor its upstream counterpart vllm-project/vllm#50496 touches any file
            under `vllm/v1/spec_decode/`, so the bug ships upstream as-is. Today nobody can run
            draft-model speculation with Apertus 1.5 without a bind-mount overlay.
          </Text>
        </Callout>
        <Callout tone="warning" title="max_model_len had to come down from 262144 to 131072">
          <Text size="small">
            With the drafter resident, job 3374717 could not reserve cache for a single full-context
            request: “To serve at least one request with the model's max seq len (262144), 28.0 GiB
            KV cache is needed, which is larger than the available KV cache memory (26.86 GiB).” No
            protocol workload exceeds 65,536 input tokens, so 131072 was applied to every arm to keep
            them comparable. Anyone copying the example at its default 262144 hits this the moment a
            drafter is added.
          </Text>
        </Callout>
        <Callout tone="warning" title="Two launch-path hazards, both worked around in `launch/`">
          <Text size="small">
            The environment toml ships a literal `{"{arch}"}` in its image path; an `sml` build that
            does not substitute it on the node makes pyxis reject the placeholder and the job dies
            seconds after start (`launch/resolve-env.sh` writes a resolved copy). And `model-launch`
            commit `4413441` renamed both the CLI flags (`--firecrest-system` → `--system`, and
            friends) and the OpenTela share mount (`/ocfbin` → `/opentelabin`) together, so an `sml`
            from the other side of that commit paired with the pinned environment toml submits fine,
            starts its container, then dies on a missing binary — a failure on the node rather than
            at submission, which is the expensive kind.
          </Text>
        </Callout>
      </Grid>
    </Stack>
  );
}

function NextSteps() {
  const steps: Array<{ title: string; body: Node }> = [
    {
      title: "Profile the draft path",
      body: (
        <Text size="small">
          A kernel-level trace of `draft-n3-tp4` against the baseline, to find where{" "}
          {span(draftCost, 1)}× a decode step goes when three 8B forward passes should cost a
          fraction of one 70B step. Hypotheses to discriminate: drafter steps not using captured or
          compiled graphs, per-step model-switch and synchronisation overhead, and the 7168-token
          budget. If that overhead is fixable, the whole conclusion changes. Highest-value item, and
          a vLLM question rather than a workload question.
        </Text>
      ),
    },
    {
      title: "Repeat the baseline — the cheapest measurement that would firm up the win",
      body: (
        <Text size="small">
          It is the only arm with one deployment and the denominator of every ratio on this canvas, so
          all quoted spreads are of the numerator alone. One extra baseline launch is now worth more
          than a third n-gram repeat, and it would also supply the baseline-versus-baseline control
          the correctness gate has never had.
        </Text>
      ),
    },
    {
      title: "Launch a baseline with async scheduling explicitly disabled",
      body: (
        <Text size="small">
          Ideally back to back with the repeat above, since both are baseline launches. It removes the
          last confound between the baseline and both speculative arms, and tests the leading
          hypothesis for why n-gram has lower TTFT than the baseline in all 12 of its cells. The
          pinned vLLM supports it through the same config field.
        </Text>
      ),
    },
    {
      title: "A genuinely long-context summarization workload",
      body: (
        <Text size="small">
          16k–64k input tokens, against the ~283 prompt tokens per request this corpus actually
          carries. Prompt lookup should do best exactly where there is a document to copy spans from,
          so n-gram's best regime is currently untested and its advantage is probably understated.
        </Text>
      ),
    },
    {
      title: "Decide a path for swiss-ai/vllm#20",
      body: (
        <Text size="small">
          Get the one line into a built image, whether through `apertus-1-5`, a dedicated branch, or
          upstream. Until then every draft-model measurement on Apertus 1.5 carries a bind-mount
          overlay in its provenance and no stock image can reproduce it.
        </Text>
      ),
    },
  ];
  return (
    <Stack gap={12}>
      <H2>What would move this forward</H2>
      <Stack gap={10}>
        {steps.map((s, i) => (
          <Row key={s.title} gap={14} align="start">
            <Text weight="semibold" tone="tertiary" style={{ minWidth: 20 }}>
              {i + 1}
            </Text>
            <Stack gap={2}>
              <Text weight="semibold">{s.title}</Text>
              {s.body}
            </Stack>
          </Row>
        ))}
      </Stack>
      <Caption>
        Two open cells a later result slots straight into: the depth sweep at draft TP=4
        (`num_speculative_tokens` 2, 5, 8) and the n-gram depth sweep. Neither is expected to change
        the sign of the draft-model result, because depth changes how much drafting happens per step
        and not the per-step overhead the decomposition isolates.
      </Caption>
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
        between the speculative arms and inflates any draft-versus-n-gram comparison. Both were
        re-read from n-gram repeat 2's own server log rather than assumed to carry over, and both
        hold identically. This is a measurement of the shipped configurations, not of speculation in
        isolation.
      </Text>
    ),
  },
  {
    title: "The baseline is the only single-deployment arm, and it is every ratio's denominator",
    tone: "warning",
    body: (
      <Text size="small">
        Screening asks for two independent deployments per arm. Both speculative arms now have them;
        the baseline has one. So it contributes no deployment-level spread, every spread quoted on
        this canvas is of the numerator alone, and a baseline repeat would widen them all — which is
        why it is now the cheapest measurement worth making, ahead of a third n-gram repeat. Two
        repeats also support a spread but not the bootstrap confidence interval over deployment
        effects that the protocol's analysis section asks for.
      </Text>
    ),
  },
  {
    title: "Different nodes, different times",
    tone: "warning",
    body: (
      <Text size="small">
        Baseline job 3391425 ran on nid007645, draft repeat 1 (3391426) on nid006633, draft repeat 2
        (3392153) and n-gram repeat 1 (3392370) on nid006687, and n-gram repeat 2 (3403354) on
        nid007500 — sequentially, because the `debug` QOS permits one running job at a time (the
        second queued with reason `QOSMaxJobs`). Node and time effects are not separated from
        `method`; the n-gram-repeat-1 / draft-repeat-2 pairing is the only node-matched one. Draft
        repeat 2 was uniformly slower than its repeat 1 and n-gram repeat 2 uniformly faster than
        its own, so the within-pair shift has no fixed direction.
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
    title: "One cell of the draft arm breaches the repeat-spread gate",
    tone: "warning",
    body: (
      <Text size="small">
        `long_context_summarization` at concurrency 1 moves{" "}
        {pct(ratioSpreads[CELLS.findIndex((c) => c.key === "summ-c1")], 1)} between the two draft
        deployments, against `maximum_repeat_spread_fraction: 0.10`. The regression's direction is
        unaffected — the draft arm returns {span(allThroughputRatios, 3)}× baseline output throughput
        across all twelve cells, far outside deployment-level variability — but that cell's effect size
        is the least settled of the six. All six n-gram cells clear the gate, worst{" "}
        {pct(Math.max(...ngramRatioSpreads), 1)}.
      </Text>
    ),
  },
  {
    title: "The correctness gate splits on the n-gram arm",
    tone: "danger",
    body: (
      <Text size="small">
        Judged against the n-gram arm's own cross-deployment control, baseline-versus-repeat-1 returns
        `within_control_envelope` and baseline-versus-repeat-2 returns `exceeds_control_envelope` on
        one statistic of four. Two deployments of one configuration disagreeing against an identical
        control means the gate's resolution is coarser than the effect it is asked to detect, so{" "}
        <Text size="small" weight="semibold">
          “the gate passes for n-gram” is not a claim this run supports
        </Text>
        . Six prompts and a worst-of-six allowance from a single control pair are not enough to
        adjudicate losslessness.
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
              "a second baseline deployment — the only arm still at one",
              "a baseline-vs-baseline correctness control, which no arm can supply",
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
    ["ngram-n3 r1", c, c.ngram] as const,
    ["ngram-n3 r2", c, c.ngram2] as const,
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
      <H2>All 30 cells as recorded</H2>
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
              : r[1] === "ngram-n3 r1" || r[1] === "ngram-n3 r2"
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
        2, green the two n-gram deployments.
      </Caption>
    </Stack>
  );
}

function Provenance() {
  const lines: Array<[string, string]> = [
    ["narrative companion", "docs/hackathon-20260915.md"],
    ["run directory", "results/smoke-screening-20260913-debug"],
    [
      "correctness artifacts",
      "results/correctness/smoke-20260913-debug (captures, exact-match comparisons, calibrated gate)",
    ],
    ["deployment failure", "results/deployment-failures/draft-n3-tp1-3392110"],
    ["baseline", "job 3391425, nid007645, replica head 172.28.51.237, stock image"],
    ["draft repeat 1", "job 3391426, nid006633, replica head 172.28.32.244, patched overlay"],
    ["draft repeat 2", "job 3392153, nid006687, replica head 172.28.33.172, patched overlay"],
    [
      "ngram repeat 1",
      "job 3392370, nid006687, replica head 172.28.33.172, stock image, prompt_lookup_min 1, prompt_lookup_max 4",
    ],
    [
      "ngram repeat 2",
      "job 3403354, nid007500, replica head 172.28.44.144, stock image, identical configuration to repeat 1",
    ],
    [
      "measurement windows",
      "2026-09-13T15:35:52Z → 16:35:24Z (baseline and draft repeat 1), 17:19Z → 17:35Z (draft repeat 2), 18:02Z → 18:09Z (n-gram repeat 1), 2026-09-14T22:12:26Z → 22:18:11Z (n-gram repeat 2)",
    ],
    ["vLLM", "a601a9d998ddeb488f0c17e8512874b116aa7658, build 0.23.1rc1.dev1029+ga601a9d99"],
    [
      "container image",
      "/capstor/store/cscs/swissai/infra01/container-images/ci/vllm_apertus_1.5_release-arm64.sqsh",
    ],
    ["model-launch", "909026a990454557f1b54d26f24ec3ad92e51e35"],
    [
      "harness",
      "apertus-bench 0.1.0, Python 3.13.9; 270413f22f41ad008a72fffae5359cb3bd03785f for the baseline and both draft repeats, a32a5e0c1c8fcb53aacf3bb305e2ad44acf5692a for ngram repeat 1, 85ccf4dcf65366cd3927f80eebc2fcf046fa1f4d for ngram repeat 2 — measurement path byte-identical across all three (per-cell `metadata.json`)",
    ],
    [
      "corpus",
      "workloads/smoke.jsonl, sha256 316fb566a18e32305b98929e34fdac515b9a43c6c7329994b319870f879952af",
    ],
    ["sampling", "temperature 0.0, top_p 1.0, seed 1 + request_index, natural EOS"],
    [
      "KV cache",
      "513,696 tokens baseline; 416,640 on both n-gram repeats; 251,472 (draft repeat 1) and 251,488 (draft repeat 2)",
    ],
    [
      "completion tokens",
      "36,148 baseline; 36,160 and 36,164 draft; 36,184 and 36,115 n-gram — agreeing within 0.2% across all five deployments",
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
      <Verdict />
      <Headline />
      <Divider />
      <PairedComparison />
      <Divider />
      <AcceptanceSection />
      <Divider />
      <StepCost />
      <Divider />
      <PlatformFindings />
      <Divider />
      <ConfigDiff />
      <Divider />
      <DraftTpOne />
      <Divider />
      <MemorySection />
      <Divider />
      <Correctness />
      <Divider />
      <DeploymentSpread />
      <Divider />
      <Caveats />
      <Divider />
      <NextSteps />
      <Divider />
      <AllCells />
      <Provenance />
    </Stack>
  );
}
