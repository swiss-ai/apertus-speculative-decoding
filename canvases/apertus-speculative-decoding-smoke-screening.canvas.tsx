/**
 * First paired measurement of speculative decoding for Apertus-v1.5-70B.
 *
 * Every number rendered here is read from the committed run artifacts of
 * `results/smoke-screening-20260913-debug` (commit 4e33323):
 *   - `analysis.csv` and each cell's `summary.json` (client-observed metrics,
 *     speculative-decoding counter deltas),
 *   - `metrics_after.prom` (`vllm:cache_config_info`) for KV-cache capacity,
 *   - `results/correctness/smoke-20260913-debug/comparison.json` (greedy gate).
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

type BaselineArm = {
  ttftP50: number;
  ttftP95: number;
  tpotP50: number;
  tpotP95: number;
  outTps: number;
  e2eP50: number;
  e2eP95: number;
  gapP50: number;
  promptTokens: number;
  completionTokens: number;
};

type DraftArm = BaselineArm & {
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
  baseline: BaselineArm;
  draft: DraftArm;
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
    labelDx: 8,
    labelDy: -8,
    baseline: {
      ttftP50: 31.01577004417777,
      ttftP95: 35.53785252152011,
      tpotP50: 14.129014411718456,
      tpotP95: 14.138498527170963,
      outTps: 70.12972149602668,
      e2eP50: 1924.3,
      e2eP95: 2736.0,
      gapP50: 14.12,
      promptTokens: 1944,
      completionTokens: 3240,
    },
    draft: {
      ttftP50: 111.62124352995306,
      ttftP95: 113.47290540579706,
      tpotP50: 31.864067355517847,
      tpotP95: 32.49081057331325,
      outTps: 31.0126732695681,
      e2eP50: 4367.7,
      e2eP95: 6133.7,
      gapP50: 90.77,
      promptTokens: 1944,
      completionTokens: 3264,
      drafts: 1128,
      draftTokens: 3384,
      acceptedTokens: 2124,
      acceptanceRate: 0.6276595744680851,
      meanAcceptanceLength: 2.882978723404255,
      perPosition: [0.7659574468085106, 0.6382978723404256, 0.4787234042553192],
    },
  },
  {
    key: "chat-c8",
    short: "chat c8",
    workload: "open_chat",
    concurrency: 8,
    labelDx: 8,
    labelDy: 14,
    baseline: {
      ttftP50: 44.3590774666518,
      ttftP95: 64.93447046959773,
      tpotP50: 14.69453933682549,
      tpotP95: 14.714398575106081,
      outTps: 432.25749510362783,
      e2eP50: 1997.6,
      e2eP95: 2870.0,
      gapP50: 14.66,
      promptTokens: 1944,
      completionTokens: 3240,
    },
    draft: {
      ttftP50: 197.11150752846152,
      ttftP95: 221.91237719962373,
      tpotP50: 32.54491677104559,
      tpotP95: 33.797205998652984,
      outTps: 190.5939564277659,
      e2eP50: 4400.5,
      e2eP95: 6509.7,
      gapP50: 91.21,
      promptTokens: 1944,
      completionTokens: 3248,
      drafts: 1122,
      draftTokens: 3366,
      acceptedTokens: 2131,
      acceptanceRate: 0.6330956625074272,
      meanAcceptanceLength: 2.8992869875222818,
      perPosition: [0.7976827094474154, 0.6550802139037433, 0.446524064171123],
    },
  },
  {
    key: "code-c1",
    short: "code c1",
    workload: "code",
    concurrency: 1,
    labelDx: 8,
    labelDy: -8,
    baseline: {
      ttftP50: 27.751603513024747,
      ttftP95: 30.49125950783491,
      tpotP50: 14.150161672427316,
      tpotP95: 14.166318852297582,
      outTps: 70.46974580539374,
      e2eP50: 5447.3,
      e2eP95: 5456.6,
      gapP50: 14.14,
      promptTokens: 2184,
      completionTokens: 9216,
    },
    draft: {
      ttftP50: 131.12756691407412,
      ttftP95: 133.49370927317068,
      tpotP50: 30.160477702255427,
      tpotP95: 31.152575448892193,
      outTps: 33.01537628400447,
      e2eP50: 11684.5,
      e2eP95: 12051.1,
      gapP50: 96.05,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 2748,
      draftTokens: 8244,
      acceptedTokens: 6492,
      acceptanceRate: 0.7874818049490538,
      meanAcceptanceLength: 3.3624454148471616,
      perPosition: [0.8864628820960698, 0.7729257641921398, 0.7030567685589519],
    },
  },
  {
    key: "code-c8",
    short: "code c8",
    workload: "code",
    concurrency: 8,
    labelDx: 8,
    labelDy: 16,
    baseline: {
      ttftP50: 44.78542983997613,
      ttftP95: 66.70471947873011,
      tpotP50: 14.700184079548514,
      tpotP95: 14.706115846099046,
      outTps: 540.6142187291132,
      e2eP50: 5676.4,
      e2eP95: 5696.3,
      gapP50: 14.69,
      promptTokens: 2184,
      completionTokens: 9216,
    },
    draft: {
      ttftP50: 218.02396804559976,
      ttftP95: 251.06144183082506,
      tpotP50: 30.069864900933467,
      tpotP95: 30.942968094105012,
      outTps: 255.17896653041808,
      e2eP50: 11733.5,
      e2eP95: 12083.6,
      gapP50: 96.23,
      promptTokens: 2184,
      completionTokens: 9216,
      drafts: 2753,
      draftTokens: 8259,
      acceptedTokens: 6490,
      acceptanceRate: 0.7858094200266376,
      meanAcceptanceLength: 3.357428260079913,
      perPosition: [0.891754449691246, 0.7686160552124954, 0.6970577551761714],
    },
  },
  {
    key: "summ-c1",
    short: "summ c1",
    workload: "long_context_summarization",
    concurrency: 1,
    labelDx: 8,
    labelDy: 16,
    baseline: {
      ttftP50: 33.5973211331293,
      ttftP95: 36.52248587459326,
      tpotP50: 14.158176855179724,
      tpotP95: 14.176949262596949,
      outTps: 70.21602782996129,
      e2eP50: 3332.3,
      e2eP95: 3651.2,
      gapP50: 14.15,
      promptTokens: 6780,
      completionTokens: 5616,
    },
    draft: {
      ttftP50: 120.28303404804319,
      ttftP95: 132.50504455063492,
      tpotP50: 32.720212998670405,
      tpotP95: 36.782325995519905,
      outTps: 29.42025630021274,
      e2eP50: 7724.2,
      e2eP95: 9501.6,
      gapP50: 93.53,
      promptTokens: 6780,
      completionTokens: 5592,
      drafts: 1932,
      draftTokens: 5796,
      acceptedTokens: 3660,
      acceptanceRate: 0.6314699792960663,
      meanAcceptanceLength: 2.8944099378881987,
      perPosition: [0.7639751552795031, 0.6211180124223602, 0.5093167701863354],
    },
  },
  {
    key: "summ-c8",
    short: "summ c8",
    workload: "long_context_summarization",
    concurrency: 8,
    labelDx: -62,
    labelDy: 4,
    baseline: {
      ttftP50: 45.67884199786931,
      ttftP95: 70.48749225214124,
      tpotP50: 14.716388219439306,
      tpotP95: 14.732173119416377,
      outTps: 493.6571058686475,
      e2eP50: 3472.0,
      e2eP95: 3822.7,
      gapP50: 14.7,
      promptTokens: 6780,
      completionTokens: 5620,
    },
    draft: {
      ttftP50: 198.44826601911336,
      ttftP95: 234.4543559011072,
      tpotP50: 31.36211582968766,
      tpotP95: 34.18852263153074,
      outTps: 211.96595167755117,
      e2eP50: 7554.0,
      e2eP95: 8952.8,
      gapP50: 91.05,
      promptTokens: 6780,
      completionTokens: 5624,
      drafts: 1967,
      draftTokens: 5901,
      acceptedTokens: 3660,
      acceptanceRate: 0.6202338586680224,
      meanAcceptanceLength: 2.8607015760040673,
      perPosition: [0.7661413319776309, 0.6014234875444839, 0.4931367564819522],
    },
  },
];

/** `vllm:cache_config_info` from each arm's `metrics_after.prom`. */
const KV_CACHE = {
  baseline: { tokens: 513696, maxConcurrency: 3.919189453125, gpuBlocks: 32106 },
  draft: { tokens: 251472, maxConcurrency: 1.9185791015625, gpuBlocks: 15717 },
};

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

const throughputRatios = CELLS.map((c) => ratio(c.draft.outTps, c.baseline.outTps));
const tpotRatios = CELLS.map((c) => ratio(c.draft.tpotP50, c.baseline.tpotP50));
const ttftRatios = CELLS.map((c) => ratio(c.draft.ttftP50, c.baseline.ttftP50));
const acceptances = CELLS.map((c) => c.draft.acceptanceRate);
const mals = CELLS.map((c) => c.draft.meanAcceptanceLength);
/**
 * Derived: cost of one draft-plus-verify step relative to one baseline decode
 * step, as (TPOT p50 × mean acceptance length) / baseline TPOT p50.
 */
const stepCostRatios = CELLS.map(
  (c) => (c.draft.tpotP50 * c.draft.meanAcceptanceLength) / c.baseline.tpotP50,
);

const span = (xs: number[], digits = 2, suffix = "") =>
  `${num(Math.min(...xs), digits)}–${num(Math.max(...xs), digits)}${suffix}`;

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
      <H1>Speculative decoding on Apertus-v1.5-70B: first paired measurement</H1>
      <Text tone="secondary">
        One 4×GH200 Clariden node per arm. Baseline `swiss-ai/Apertus-v1.5-70B` at tensor-parallel 4
        against the same target with an `Apertus-v1.5-8B` draft model,
        `num_speculative_tokens=3`, `draft_tensor_parallel_size=4`. Three workloads × two
        concurrencies × two arms = 12 cells, 8 warmup then 24 measured requests each, all with a
        1.000 success rate.
      </Text>
      <Row gap={6} wrap>
        <Pill size="sm">Clariden · partition debug</Pill>
        <Pill size="sm">2026-09-13, 15:35–16:35 UTC</Pill>
        <Pill size="sm">jobs 3391425 / 3391426</Pill>
        <Pill size="sm">smoke corpus, 6 prompts</Pill>
        <Pill size="sm" tone="warning">
          1 deployment repeat per arm
        </Pill>
        <Pill size="sm">commit 4e33323</Pill>
      </Row>
    </Stack>
  );
}

function Headline() {
  return (
    <Grid columns="1.5fr 1fr" gap={20} align="start">
      <Stack gap={16}>
        <Text weight="semibold" style={{ fontSize: 17, lineHeight: 1.45 }}>
          The drafter's tokens are accepted 62–79% of the time, and the speculative arm is still
          slower in all six cells. Acceptance is not the binding constraint in this configuration.
        </Text>
        <Grid columns={3} gap={16}>
          <Stat
            value={`${span(throughputRatios)}×`}
            label="Output throughput vs baseline (all 6 cells)"
            tone="danger"
          />
          <Stat value={`${span(tpotRatios)}×`} label="TPOT p50 vs baseline" tone="danger" />
          <Stat value={`${span(ttftRatios)}×`} label="TTFT p50 vs baseline" tone="danger" />
          <Stat value={span(acceptances, 3)} label="Acceptance rate" tone="success" />
          <Stat
            value={span(mals)}
            label="Mean acceptance length (of a possible 4)"
            tone="success"
          />
          <Stat
            value={`−${num((1 - KV_CACHE.draft.tokens / KV_CACHE.baseline.tokens) * 100, 1)}%`}
            label="KV-cache tokens after hosting the drafter"
            tone="danger"
          />
        </Grid>
      </Stack>
      <Card>
        <CardHeader trailing="held constant">configuration</CardHeader>
        <CardBody>
          <Stack gap={6}>
            {([
              ["target", "swiss-ai/Apertus-v1.5-70B, TP=4"],
              ["draft", "swiss-ai/Apertus-v1.5-8B, TP=4"],
              ["speculative tokens", "3"],
              ["max_model_len", "131072"],
              ["gpu_memory_utilization", "0.8"],
              ["sampling", "temperature 0.0, top_p 1.0, natural EOS"],
              ["per cell", "8 warmup + 24 measured, closed loop"],
              ["load generator", "on-cluster, direct to replica node IP"],
            ] as Array<[string, string]>).map(([k, v]) => (
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

function PairedComparison() {
  const [metricId, setMetricId] = useCanvasState<MetricId>("pairedMetric", "tpotP50");
  const metric = METRICS[metricId];
  const baselineData = CELLS.map((c) => round(c.baseline[metricId], 2));
  const draftData = CELLS.map((c) => round(c.draft[metricId], 2));
  const higherIsWorse = metricId !== "outTps";

  return (
    <Stack gap={12}>
      <H2>Paired per-cell comparison</H2>
      <Text tone="secondary">
        The same corpus, request order, and client drive both arms. Pick a metric; the regression has
        the same sign in every workload/concurrency cell.
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
          { name: "baseline (method=none)", data: baselineData, tone: "info" },
          { name: "draft-n3-tp4 (method=draft_model)", data: draftData, tone: "danger" },
        ]}
        valueSuffix={` ${metric.unit}`}
        height={300}
      />
      <Caption>
        y: {metric.axis} · x: workload × concurrency cell (`chat` = open_chat, `summ` =
        long_context_summarization; `c1`/`c8` = concurrency) · {higherIsWorse ? "higher" : "lower"}{" "}
        is worse · client-observed, 24 measured requests per cell · source: `analysis.csv`,
        `smoke-screening-20260913-debug`
      </Caption>

      <Divider />

      <H3>Speculative arm as a fraction of baseline output throughput</H3>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "draft-n3-tp4 ÷ baseline output throughput",
            data: throughputRatios.map((r) => round(r, 3)),
            tone: "danger",
          },
        ]}
        valueSuffix="×"
        horizontal
        height={240}
      />
      <Caption>
        x: output-throughput ratio (draft ÷ baseline, dimensionless; 1.00× = parity, none of the six
        cells reaches 0.48×) · y: workload × concurrency cell · this is the
        `speedup_vs_baseline` column of `analysis.csv`
      </Caption>
    </Stack>
  );
}

/** Acceptance rate (x) against measured output-throughput ratio (y), 6 draft cells. */
function AcceptanceScatter() {
  const theme = useHostTheme();
  const width = 620;
  const height = 300;
  const m = { top: 14, right: 22, bottom: 46, left: 62 };
  const xMin = 0.6;
  const xMax = 0.82;
  const yMin = 0.4;
  const yMax = 0.5;
  const px = (v: number) => m.left + ((v - xMin) / (xMax - xMin)) * (width - m.left - m.right);
  const py = (v: number) => height - m.bottom - ((v - yMin) / (yMax - yMin)) * (height - m.top - m.bottom);
  const xTicks = [0.6, 0.65, 0.7, 0.75, 0.8];
  const yTicks = [0.4, 0.42, 0.44, 0.46, 0.48, 0.5];

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
            <text
              x={m.left - 8}
              y={py(t) + 4}
              textAnchor="end"
              fontSize={10}
              fill={theme.text.tertiary}
            >
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
        <line x1={m.left} x2={m.left} y1={m.top} y2={height - m.bottom} stroke={theme.stroke.primary} />
        {CELLS.map((c) => {
          const x = px(c.draft.acceptanceRate);
          const y = py(ratio(c.draft.outTps, c.baseline.outTps));
          return (
            <g key={c.key}>
              <circle cx={x} cy={y} r={5} fill={theme.accent.primary} />
              <text
                x={x + c.labelDx}
                y={y + c.labelDy}
                fontSize={10}
                fill={theme.text.secondary}
              >
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
      <Caption>
        Six draft-arm cells, one deployment repeat each. The y-axis is zoomed to 0.40–0.50×: parity
        (1.00×) is far above the top of this plot. Acceptance spans {span(acceptances, 3)} while the
        throughput ratio spans {span(throughputRatios, 3)}× — within this run, higher acceptance
        coincides with a marginally smaller loss, never a gain. Source: per-cell `summary.json`
        counter deltas.
      </Caption>
    </Stack>
  );
}

function AcceptanceSection() {
  return (
    <Stack gap={12}>
      <H2>Acceptance is high, and it does not buy throughput</H2>
      <Text tone="secondary">
        vLLM accepts a mean of {span(mals)} tokens per verification step out of a possible 4
        (3 drafted + 1 bonus). If a speculative step cost the same as a baseline decode step, that
        would be the speedup. The measured ratio is {span(throughputRatios)}×, which puts one
        draft-plus-verify step at {span(stepCostRatios)}× the cost of one baseline decode step
        (derived: TPOT p50 × mean acceptance length ÷ baseline TPOT p50). The measured stream-event
        gap of ~91–96 ms in the draft arm, against ~14 ms in the baseline, is the same quantity
        observed directly at the client.
      </Text>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "Upper bound implied by mean acceptance length (derived)",
            data: mals.map((v) => round(v, 2)),
            tone: "success",
          },
          {
            name: "Measured output-throughput ratio",
            data: throughputRatios.map((r) => round(r, 3)),
            tone: "danger",
          },
        ]}
        valueSuffix="×"
        height={280}
      />
      <Caption>
        y: throughput relative to baseline (×, dimensionless) · x: workload × concurrency cell · the
        green series is <Text size="small" italic tone="tertiary">derived</Text>, not measured: it is
        the mean acceptance length, i.e. the speedup that would follow only if a draft-plus-verify
        step were as cheap as one baseline decode step. It is an upper bound and this run shows it is
        far from attained.
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
                name: "Mean acceptance length (tokens per verification step)",
                data: mals.map((v) => round(v, 3)),
                tone: "success",
              },
            ]}
            valueSuffix=" tok"
            height={244}
          />
          <Caption>
            y: tokens accepted per verification step, `1 + accepted/drafts`, including the bonus
            token (maximum 4 at `num_speculative_tokens=3`) · x: workload × concurrency cell ·
            `code` is highest in both concurrencies, the direction hypothesis H2 predicts.
          </Caption>
        </Stack>
      </Grid>

      <Divider />

      <H3>Per-position acceptance</H3>
      <BarChart
        categories={CATEGORIES}
        series={[
          {
            name: "overall acceptance rate",
            data: acceptances.map((v) => round(v * 100, 1)),
            tone: "neutral",
          },
          { name: "position 0", data: CELLS.map((c) => round(c.draft.perPosition[0] * 100, 1)) },
          { name: "position 1", data: CELLS.map((c) => round(c.draft.perPosition[1] * 100, 1)) },
          { name: "position 2", data: CELLS.map((c) => round(c.draft.perPosition[2] * 100, 1)) },
        ]}
        valueSuffix="%"
        height={300}
      />
      <Caption>
        y: acceptance (% of drafted tokens at that position) · x: workload × concurrency cell ·
        positions are vLLM's 0/1/2 within each 3-token proposal · acceptance decays with position in
        every cell, and `code` decays least (88.6→70.3% at c1) · source:
        `vllm:spec_decode_num_accepted_tokens_per_pos_total` deltas in each cell's `summary.json`.
      </Caption>

      <Card>
        <CardHeader trailing="draft arm only">speculative-decoding counter deltas</CardHeader>
        <CardBody style={{ padding: 0 }}>
          <Table
            framed={false}
            headers={[
              "Workload",
              "Conc.",
              "Drafts",
              "Draft tokens",
              "Accepted",
              "Acceptance",
              "Mean acc. length",
              "Per-position (0/1/2)",
            ]}
            columnAlign={["left", "right", "right", "right", "right", "right", "right", "right"]}
            rows={CELLS.map((c) => [
              c.workload,
              c.concurrency,
              c.draft.drafts.toLocaleString("en-US"),
              c.draft.draftTokens.toLocaleString("en-US"),
              c.draft.acceptedTokens.toLocaleString("en-US"),
              num(c.draft.acceptanceRate, 3),
              num(c.draft.meanAcceptanceLength, 2),
              c.draft.perPosition.map((p) => num(p, 3)).join(" / "),
            ])}
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

function MemorySection() {
  const kvRatio = KV_CACHE.draft.tokens / KV_CACHE.baseline.tokens;
  return (
    <Stack gap={12}>
      <H2>What hosting the drafter costs in KV cache</H2>
      <Grid columns="1fr 1fr" gap={20} align="start">
        <Stack gap={8}>
          <BarChart
            categories={["baseline", "draft-n3-tp4"]}
            series={[
              {
                name: "Advertised KV-cache capacity",
                data: [KV_CACHE.baseline.tokens, KV_CACHE.draft.tokens],
              },
            ]}
            valueSuffix=" tok"
            height={220}
          />
          <Caption>
            y: KV-cache capacity (tokens) · x: arm · both arms ran at
            `gpu_memory_utilization=0.8` on 4 GH200 · source: `kv_cache_size_tokens` in
            `vllm:cache_config_info`, `metrics_after.prom`.
          </Caption>
        </Stack>
        <Stack gap={12}>
          <Grid columns={2} gap={16}>
            <Stat
              value={`${num(KV_CACHE.baseline.maxConcurrency)}×`}
              label="baseline max concurrency at 131072 tok/request"
            />
            <Stat
              value={`${num(KV_CACHE.draft.maxConcurrency)}×`}
              label="draft-n3-tp4 max concurrency at 131072 tok/request"
              tone="danger"
            />
            <Stat
              value={KV_CACHE.baseline.gpuBlocks.toLocaleString("en-US")}
              label="baseline GPU blocks (block_size 16)"
            />
            <Stat
              value={KV_CACHE.draft.gpuBlocks.toLocaleString("en-US")}
              label="draft-n3-tp4 GPU blocks (block_size 16)"
              tone="danger"
            />
          </Grid>
          <Text tone="secondary">
            Loading the 8B drafter costs {pct(1 - kvRatio, 1)} of the target's advertised KV-cache
            capacity. At the configured `max_model_len` the server advertises capacity for
            fewer than two full-length requests, which is a capacity result independent of any
            latency measurement — and it applies even if the latency regression above turns out to
            be a configuration artefact.
          </Text>
        </Stack>
      </Grid>
    </Stack>
  );
}

const CAVEATS: Array<{ title: string; tone: "warning" | "danger"; body: Node }> = [
  {
    title: "The two arms did not run the same image",
    tone: "danger",
    body: (
      <Text size="small">
        The draft arm ran with `patches/vllm-apertus-image-token.patch` bind-mounted over
        `/workspace/vllm/vllm/v1/spec_decode/llm_base_proposer.py` in the pinned image, because that
        image reads `image_token_index` off the target config and Apertus 1.5 only defines
        `image_token_id`, so the drafter cannot load without it. The baseline used the stock image.
        The fix is open upstream as swiss-ai/vllm#20 and unmerged.
      </Text>
    ),
  },
  {
    title: "Two serving properties are confounded with method",
    tone: "danger",
    body: (
      <Text size="small">
        vLLM auto-disables async scheduling for `draft_model` speculation (the baseline logs
        `Asynchronous scheduling is enabled`, the draft arm logs disabled), and it shrinks the
        per-step token budget from `max_num_batched_tokens=8192` to
        `max_num_scheduled_tokens=7168`. Both penalise the speculative arm independently of
        acceptance and cannot be held constant in this revision, so this is a measurement of the
        shipped configuration, not of speculation in isolation.
      </Text>
    ),
  },
  {
    title: "The greedy exact-match gate failed: 0 of 6",
    tone: "danger",
    body: (
      <Text size="small">
        `require_exact_greedy_smoke_match` did not pass on any prompt. Each pair agrees for the
        first 297–564 characters and then diverges into a different but equally fluent
        continuation, with completion-token counts within 3. That is consistent with
        floating-point nondeterminism across nodes, batch shapes, and token budgets rather than a
        rejection-sampling bug — but it is unconfirmed, and this run does not clear the gate. A
        same-node, same-batch-shape re-check is needed before any correctness claim.
      </Text>
    ),
  },
  {
    title: "One deployment repeat per arm",
    tone: "warning",
    body: (
      <Text size="small">
        The protocol's independent unit is a deployment repeat and screening asks for two; a single
        `debug` hour per arm allowed one. There is therefore no deployment-level confidence
        interval anywhere on this canvas. Six cells agreeing in direction is suggestive, not a
        substitute.
      </Text>
    ),
  },
  {
    title: "Different nodes, different times",
    tone: "warning",
    body: (
      <Text size="small">
        Baseline job 3391425 ran on nid007645, the draft job 3391426 on nid006633, sequentially,
        because the `debug` QOS permits one running job at a time (the second queued with reason
        `QOSMaxJobs`). Node and time effects are not separated from `method`.
      </Text>
    ),
  },
  {
    title: "The prefill-heavy regime is untested",
    tone: "warning",
    body: (
      <Text size="small">
        The smoke corpus' "long context" stratum records 6,780 prompt tokens across the 24 requests
        of a cell, about 283 tokens per request — far below the 16k–64k documents the
        research corpus specifies. These `long_context_summarization` rows therefore measure almost
        no prefill, which is the regime where speculation has the most room to help TTFT.
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
        measured regression, and one is an outright failed gate.
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
              "the n-gram (model-free) arm",
              "speculative depths other than 3; draft TP=1",
              "an `ignore_eos` control, so output lengths are not fixed",
              "the open-loop serving-capacity / sustainable-rate search",
              "GPU utilization, power, SM activity, communication metrics",
              "deployment-level confidence intervals (one repeat per arm)",
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
        Request-level percentiles in the tables are reported as recorded, but the 24 requests per
        cell cycle through only 2 prompts per stratum, so those distributions are narrow by
        construction and the samples are not independent.
      </Caption>
    </Stack>
  );
}

function AllCells() {
  const rows = CELLS.flatMap((c) => [
    ["baseline", c, c.baseline] as const,
    ["draft-n3-tp4", c, c.draft] as const,
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
      <H2>All 12 cells as recorded</H2>
      <Table
        headers={[
          "Cell",
          "Arm",
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
        rowTone={rows.map((r) => (r[1] === "draft-n3-tp4" ? "danger" : undefined))}
        rows={rows}
        striped
      />
      <Caption>
        TTFT is request start to the first non-empty streamed payload; TPOT is
        `(E2E − TTFT)/(completion_tokens − 1)` per request; output throughput is completion tokens
        over whole-cell wall time. Stream gap is the interval between non-empty SSE payloads and is
        a diagnostic, not token-level latency — speculation delivers several accepted tokens in one
        event, which is why the draft arm's ~91–96 ms gap sits next to a ~30–33 ms TPOT while the
        baseline's gap and TPOT coincide at ~14 ms. Rows tinted red are the speculative arm.
      </Caption>
    </Stack>
  );
}

function Provenance() {
  const lines: Array<[string, string]> = [
    ["run directory", "results/smoke-screening-20260913-debug (commit 4e33323)"],
    ["correctness captures", "results/correctness/smoke-20260913-debug"],
    ["baseline", "job 3391425, nid007645, replica head 172.28.51.237, stock image"],
    ["draft-n3-tp4", "job 3391426, nid006633, replica head 172.28.32.244, patched overlay"],
    ["measurement window", "2026-09-13T15:35:52Z → 2026-09-13T16:35:24Z"],
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
      <PairedComparison />
      <Divider />
      <AcceptanceSection />
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
