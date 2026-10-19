import { expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { mergeDeep } from "remeda";

import { groups, providers, syncProvider, type ExistingModel, type SyncedModel, type SyncProvider } from "../src/sync/index.js";
import { MissingReasoningOptionsError } from "../src/sync/missing-reasoning-options.js";
import {
  API_ENDPOINT,
  buildMunitoModel,
  fetchMunitoModels,
  munito,
  parseMunitoModels,
  reasoningOptions,
  resolveMunitoBaseModel,
  type MunitoModel,
} from "../src/sync/providers/munito.js";
import { modelMetadata } from "../src/sync/providers/openrouter.js";

const REPO = path.join(import.meta.dirname, "..", "..", "..");
const SHIPPED_DIR = path.join(REPO, "providers", "munito", "models");
const LAB_DIR = path.join(REPO, "models");

interface RowOptions {
  region?: string;
  tier?: string;
  input?: string[];
  context?: number;
  output?: number;
  prompt?: string;
  completion?: string;
  cached?: string;
  reasoning?: boolean;
  effort?: string[];
  switches?: string[];
  tools?: boolean;
  structured?: boolean;
  temperature?: boolean;
}

// One row of GET https://api.munito.ai/inference/v1/models/openrouter, trimmed
// to the fields the sync reads.
function row(id: string, options: RowOptions = {}) {
  const {
    region = "global",
    input = ["text"],
    context = 262_144,
    output = 235_929,
    prompt = "0.00000023",
    completion = "0.0000025",
    reasoning = true,
    tools = true,
    structured = true,
    temperature = true,
  } = options;
  const parameters: Record<string, unknown> = {
    max_tokens: { type: "integer", min: 1, max: output, unit: "token" },
  };
  if (temperature) parameters.temperature = { type: "range", min: 0, max: 2 };
  if (tools) parameters.tools = { type: "boolean" };
  if (structured) parameters.structured_outputs = { type: "boolean" };
  if (reasoning) parameters.reasoning = { type: "boolean" };
  if (options.effort !== undefined) parameters.reasoning_effort = { type: "enum", values: options.effort };
  const textPricing = [
    { type: "prompt", unit: "token", cost_usd: prompt },
    ...options.cached === undefined ? [] : [{ type: "cached_prompt", unit: "token", cost_usd: options.cached }],
  ];
  return {
    schema_version: "2.4",
    id,
    deployment_region: region,
    ...options.tier === undefined ? {} : { service_tier: options.tier },
    input_modalities: input.map((type) =>
      type === "text"
        ? { type, supported_inputs: { max_context_length: { value: context, unit: "token" } }, pricing: textPricing }
        : { type, supported_inputs: { sources: { type: "enum", values: ["url", "base64"] } } }
    ),
    output_modalities: [{
      type: "text",
      max_length: { value: output, unit: "token" },
      streaming: true,
      pricing: [{ type: "completion", unit: "token", cost_usd: completion }],
      supported_parameters: parameters,
    }],
    ...options.switches === undefined ? {} : {
      passthrough_parameters: {
        chat_template_kwargs: {
          type: "object",
          properties: Object.fromEntries(options.switches.map((name) => [name, { type: "boolean" }])),
        },
      },
    },
  };
}

// The live feed on 2026-10-19 for the models the sync creates or skips.
const QWEN38 = row("qwen/qwen3.8-27b", {
  input: ["text", "image", "video"],
  cached: "0.00000005",
  effort: ["low", "medium", "xhigh"],
  switches: ["enable_thinking", "preserve_thinking"],
});
const KOLIBRI = row("aleph-alpha/kolibri-1", {
  context: 1_048_576,
  output: 943_718,
  prompt: "0.00000013",
  completion: "0.00000051",
  cached: "0.000000078",
  effort: ["none", "low", "medium", "high"],
  switches: ["enable_thinking"],
});
const QWEN36 = row("qwen/qwen3.6-35b-a3b", {
  prompt: "0.00000013",
  completion: "0.00000051",
  switches: ["enable_thinking"],
});
const GLM = row("zai-org/glm-5.3-flash", {
  input: ["text", "image", "video"],
  context: 1_048_576,
  output: 943_718,
  effort: ["low", "high", "max"],
  switches: ["clear_thinking"],
});
const DEEPSEEK = row("deepseek-ai/deepseek-v4-flash", {
  context: 1_048_576,
  output: 943_718,
  prompt: "0.0000001",
  completion: "0.00000025",
  cached: "0.00000002",
  effort: ["low", "high", "max"],
  switches: ["thinking"],
});
const MISTRAL = row("mistralai/mistral-small-4-119b-2603", { effort: ["none", "high"] });
const CYGNET = row("blockbrain/cygnet", { context: 16_384, output: 1, switches: ["enable_thinking"] });

function parsed(raw: ReturnType<typeof row>): MunitoModel {
  return parseMunitoModels({ data: [raw] })[0]!;
}

function translate(raw: ReturnType<typeof row>, authored?: ExistingModel) {
  return munito.translateModel(parsed(raw), { existing: () => authored, authored: () => authored });
}

// The provider file merged over its lab file, as consumers read it.
function resolved(model: SyncedModel): ExistingModel {
  const base = "base_model" in model ? model.base_model : undefined;
  const { benchmarks: _benchmarks, weights: _weights, links: _links, ...lab } = modelMetadata(base!);
  return mergeDeep(lab, model as Record<string, unknown>) as ExistingModel;
}

test("keeps only the standard-tier rows of the global host", () => {
  const feed = {
    data: [
      QWEN38,
      row("qwen/qwen3.8-27b", { tier: "flex", prompt: "0.00000012" }),
      row("qwen/qwen3.8-27b", { tier: "priority", prompt: "0.00000041" }),
      row("qwen/qwen3.8-27b@eu", { region: "EU" }),
      row("qwen/qwen3.8-27b@eu", { region: "EU", tier: "flex" }),
      row("qwen/qwen3.8-27b@us", { region: "US" }),
    ],
  };
  expect(parseMunitoModels(feed).map((model) => munito.sourceID(model))).toEqual(["qwen/qwen3.8-27b"]);
});

test("refuses a feed with no global standard-tier row instead of deleting every file", () => {
  expect(() => parseMunitoModels({ data: [] })).toThrow("refusing a destructive sync");
  expect(() => parseMunitoModels({ data: [row("qwen/qwen3.8-27b@eu", { region: "EU" })] }))
    .toThrow("refusing a destructive sync");
});

test("rejects an empty or non-numeric price instead of writing a free model", () => {
  expect(() => parseMunitoModels({ data: [row("qwen/qwen3.8-27b", { prompt: "" })] })).toThrow();
  expect(() => parseMunitoModels({ data: [row("qwen/qwen3.8-27b", { completion: "n/a" })] })).toThrow();
});

test("converts per-token prices to USD per 1M tokens and rounds away the float error", () => {
  expect(Number("0.00000023") * 1e6).not.toBe(0.23);
  expect(translate(QWEN38)?.model.cost).toEqual({ input: 0.23, output: 2.5, cache_read: 0.05 });
  expect(translate(KOLIBRI)?.model.cost).toEqual({ input: 0.13, output: 0.51, cache_read: 0.078 });
  expect(translate(QWEN36)?.model.cost).toEqual({ input: 0.13, output: 0.51 });
});

test("resolves wire IDs to the lab file each shipped entry points at", () => {
  expect(resolveMunitoBaseModel("qwen/qwen3.8-27b")).toBe("alibaba/qwen3.8-27b");
  expect(resolveMunitoBaseModel("zai-org/glm-5.3-flash")).toBe("zhipuai/glm-5.3-flash");
  expect(resolveMunitoBaseModel("deepseek-ai/deepseek-v4-flash")).toBe("deepseek/deepseek-v4-flash");
  expect(resolveMunitoBaseModel("mistralai/mistral-small-4-119b-2603")).toBe("mistral/mistral-small-2603");
  expect(resolveMunitoBaseModel("aleph-alpha/kolibri-1")).toBe("aleph-alpha/kolibri-1");
  expect(resolveMunitoBaseModel("blockbrain/cygnet")).toBeUndefined();
});

test("a model with no lab file is skipped and reported for a missing-model issue", () => {
  expect(translate(CYGNET)).toBeUndefined();
  expect(munito.missingModelID(parsed(CYGNET))).toBe("blockbrain/cygnet");
  expect(munito.skippedNotice(["blockbrain/cygnet"])[0]).toContain("`blockbrain/cygnet`");
  expect(munito.skippedNotice([])).toEqual([]);
});

test("an authored base_model wins over the resolver", () => {
  const authored = { base_model: "alibaba/qwen3.8-27b" } as ExistingModel;
  const translated = translate(row("blockbrain/cygnet", { effort: ["low"] }), authored);
  expect(translated?.model).toMatchObject({ base_model: "alibaba/qwen3.8-27b" });
});

test("writes only the limits and modalities that differ from the lab file", () => {
  const qwen38 = translate(QWEN38)!.model;
  expect(qwen38).not.toHaveProperty("modalities");
  expect(qwen38).not.toHaveProperty("attachment");
  expect(resolved(qwen38).limit).toMatchObject({ context: 262_144, output: 235_929 });

  // The lab file lists image, video and audio input; the host serves text.
  const qwen36 = translate(QWEN36)!.model;
  expect(qwen36).toMatchObject({ attachment: false, modalities: { input: ["text"] } });
  expect(resolved(qwen36).modalities).toEqual({ input: ["text"], output: ["text"] });

  // The lab file lists PDF input; the host serves text, image and video.
  expect(resolved(translate(GLM)!.model).modalities?.input).toEqual(["text", "image", "video"]);
  expect(resolved(translate(DEEPSEEK)!.model).limit).toMatchObject({ context: 1_048_576, output: 943_718 });
});

test("copies the reasoning controls that the host accepts", () => {
  // A separate on/off switch and graded effort without `none`: toggle + effort.
  expect(reasoningOptions(parsed(QWEN38))).toEqual([
    { type: "toggle" },
    { type: "effort", values: ["low", "medium", "xhigh"] },
  ]);
  expect(reasoningOptions(parsed(DEEPSEEK))).toEqual([
    { type: "toggle" },
    { type: "effort", values: ["low", "high", "max"] },
  ]);
  // `none` already disables reasoning, so the switch is not a second toggle.
  expect(reasoningOptions(parsed(KOLIBRI))).toEqual([{ type: "effort", values: ["none", "low", "medium", "high"] }]);
  expect(reasoningOptions(parsed(MISTRAL))).toEqual([{ type: "effort", values: ["none", "high"] }]);
  // A binary switch alone.
  expect(reasoningOptions(parsed(QWEN36))).toEqual([{ type: "toggle" }]);
  // clear_thinking removes earlier reasoning from the prompt; it does not disable thinking.
  expect(reasoningOptions(parsed(GLM))).toEqual([{ type: "effort", values: ["low", "high", "max"] }]);
});

test("names the wire path of each control in the leading comment", () => {
  expect(translate(DEEPSEEK)?.header).toBe(
    `# Source: ${API_ENDPOINT} (public), standard tier on the global host.\n`
      + "# Toggle: chat_template_kwargs.thinking = true|false\n"
      + "# Effort: reasoning_effort = low|high|max\n",
  );
  expect(translate(QWEN36)?.header).toContain("# Toggle: chat_template_kwargs.enable_thinking = true|false\n");
  expect(translate(KOLIBRI)?.header).toContain("none disables reasoning");
});

test("a reasoning model with no control in the feed waits for a human", () => {
  expect(() => translate(row("qwen/qwen3.8-27b"))).toThrow(MissingReasoningOptionsError);
  expect(() => translate(row("qwen/qwen3.8-27b", { effort: ["turbo"] }))).toThrow(MissingReasoningOptionsError);
});

test("keeps the authored interleaved field and status", () => {
  const authored = {
    base_model: "alibaba/qwen3.8-27b",
    interleaved: { field: "reasoning_details" },
    status: "deprecated",
  } as ExistingModel;
  expect(translate(QWEN38, authored)?.model).toMatchObject({
    interleaved: { field: "reasoning_details" },
    status: "deprecated",
  });
  expect(translate(QWEN38)?.model).toMatchObject({ interleaved: { field: "reasoning_content" } });
});

test("fetches the feed without auth and fails on an error status", async () => {
  const calls: unknown[][] = [];
  const ok = (async (...args: unknown[]) => {
    calls.push(args);
    return new Response(JSON.stringify({ data: [QWEN38] }));
  }) as unknown as typeof fetch;
  expect(await fetchMunitoModels(ok)).toEqual({ data: [QWEN38] });
  expect(calls).toEqual([[API_ENDPOINT]]);
  const down = (async () => new Response("", { status: 503, statusText: "Service Unavailable" })) as unknown as typeof fetch;
  await expect(fetchMunitoModels(down)).rejects.toThrow("Munito models request failed: 503 Service Unavailable");
});

test("is registered in the direct group", () => {
  expect(providers.munito).toBe(munito);
  expect(groups.direct).toContain("munito");
});

// Every shipped file, and the feed row that the host must serve for the sync
// to keep the file unchanged. The rows come from the files, so an hourly repricing cannot
// make this test stale.
const SHIPPED = shippedFiles(SHIPPED_DIR);

function shippedFiles(root: string, dir = ""): Array<{ file: string; text: string; row: ReturnType<typeof row> }> {
  return readdirSync(path.join(root, dir), { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) return shippedFiles(root, file);
    const text = readFileSync(path.join(root, file), "utf8");
    const model = resolved(Bun.TOML.parse(text) as SyncedModel);
    const options = model.reasoning_options ?? [];
    const effort = options.find((option) => option.type === "effort");
    const toggle = /chat_template_kwargs\.(\w+)/.exec(text)?.[1];
    return [{
      file,
      text,
      row: row(file.slice(0, -".toml".length), {
        input: model.modalities!.input,
        context: model.limit!.context,
        output: model.limit!.output,
        prompt: perToken(model.cost!.input),
        completion: perToken(model.cost!.output),
        cached: model.cost!.cache_read === undefined ? undefined : perToken(model.cost!.cache_read),
        reasoning: model.reasoning,
        effort: effort?.type === "effort" ? effort.values.map(String) : undefined,
        switches: toggle === undefined ? undefined : [toggle],
        tools: model.tool_call,
        structured: model.structured_output,
        temperature: model.temperature,
      }),
    }];
  });
}

function perToken(usdPerMillion: number) {
  return (usdPerMillion / 1e6).toFixed(12).replace(/0+$/, "");
}

// A temporary checkout: the Munito files plus the lab files they point at, so
// the runner resolves base_model exactly as it does in the repo.
async function inRoot(run: (modelsDir: string) => Promise<void>) {
  const root = await mkdtemp(path.join(tmpdir(), "models-dev-munito-"));
  const modelsDir = path.join(root, "providers", "munito", "models");
  try {
    for (const { file, text } of SHIPPED) {
      await mkdir(path.dirname(path.join(modelsDir, file)), { recursive: true });
      await writeFile(path.join(modelsDir, file), text);
      const base = (Bun.TOML.parse(text) as ExistingModel).base_model!;
      await mkdir(path.join(root, "models", path.dirname(base)), { recursive: true });
      await copyFile(path.join(LAB_DIR, `${base}.toml`), path.join(root, "models", `${base}.toml`));
    }
    await run(modelsDir);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

function munitoAt(modelsDir: string, source: () => unknown): SyncProvider<MunitoModel> {
  return { ...munito, modelsDir, async fetchModels() { return source(); } };
}

test("the shipped files are a fixed point", async () => {
  expect(SHIPPED.length).toBeGreaterThan(0);
  await inRoot(async (modelsDir) => {
    const provider = munitoAt(modelsDir, () => ({ data: SHIPPED.map((shipped) => shipped.row) }));
    for (const result of [await syncProvider(provider), await syncProvider(provider)]) {
      expect(result).toMatchObject({ created: 0, updated: 0, deleted: 0, unchanged: SHIPPED.length, files: [] });
    }
    for (const { file, text } of SHIPPED) {
      expect(await readFile(path.join(modelsDir, file), "utf8")).toBe(text);
    }
  });
});

test("a model that leaves the feed is deleted, and an empty feed deletes nothing", async () => {
  await inRoot(async (modelsDir) => {
    const [first, ...rest] = SHIPPED;
    await expect(syncProvider(munitoAt(modelsDir, () => ({ data: [] })))).rejects.toThrow("refusing a destructive sync");
    expect(await Bun.file(path.join(modelsDir, first!.file)).exists()).toBe(true);

    const result = await syncProvider(munitoAt(modelsDir, () => ({ data: rest.map((shipped) => shipped.row) })));
    expect(result).toMatchObject({ created: 0, updated: 0, deleted: 1 });
    expect(await Bun.file(path.join(modelsDir, first!.file)).exists()).toBe(false);
  });
});
