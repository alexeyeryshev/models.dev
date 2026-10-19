import { z } from "zod";

import type { ExistingModel, SyncProvider, SyncedFullModel, SyncedModel } from "../index.js";
import { MissingReasoningOptionsError } from "../missing-reasoning-options.js";
import { factorBaseModel, resolveModelMetadataBaseModel } from "./openrouter.js";

// Munito's public listing in OpenRouter's provider format: one row for each
// model, region and service tier, with the prices, limits, modalities and
// request parameters that the host accepts. It needs no key.
export const API_ENDPOINT = "https://api.munito.ai/inference/v1/models/openrouter";

// Wire IDs are Hugging Face organisation slugs. These organisations keep their
// lab files under another directory name.
const LAB_PREFIXES: Record<string, string> = {
  "deepseek-ai": "deepseek",
  "perplexity-ai": "perplexity",
};

// Wire IDs whose lab file has another model slug.
const BASE_MODEL_EXCEPTIONS: Record<string, string> = {
  "mistralai/mistral-small-4-119b-2603": "mistral/mistral-small-2603",
};

// chat_template_kwargs switches that enable and disable thinking. GLM's
// clear_thinking only removes earlier reasoning from the prompt, so it is not one.
const THINKING_SWITCHES = ["enable_thinking", "thinking"] as const;

const EFFORT_VALUES = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"] as const;
type EffortValue = (typeof EFFORT_VALUES)[number];

// Per-token USD as a decimal string. An empty or non-numeric string must stop
// the run, so that the sync never writes a free model.
const Price = z
  .object({
    type: z.string(),
    unit: z.string(),
    cost_usd: z.string().regex(/^\d+(\.\d+)?(e[-+]?\d+)?$/i, "munito price is not a non-negative number"),
  })
  .passthrough();

const TokenLength = z.object({ value: z.number().int().positive() }).passthrough();

const Parameter = z.object({ type: z.string(), values: z.array(z.string()).optional() }).passthrough();

const InputModality = z
  .object({
    type: z.string(),
    supported_inputs: z.object({ max_context_length: TokenLength.optional() }).passthrough().optional(),
    pricing: z.array(Price).optional(),
  })
  .passthrough();

const OutputModality = z
  .object({
    type: z.string(),
    max_length: TokenLength.optional(),
    pricing: z.array(Price).optional(),
    supported_parameters: z.record(z.string(), Parameter).default({}),
  })
  .passthrough();

export const MunitoModel = z
  .object({
    id: z.string().min(1),
    deployment_region: z.string(),
    service_tier: z.string().optional(),
    input_modalities: z.array(InputModality).min(1),
    output_modalities: z.array(OutputModality).min(1),
    passthrough_parameters: z
      .object({
        chat_template_kwargs: z
          .object({ properties: z.record(z.string(), Parameter).default({}) })
          .passthrough()
          .optional(),
      })
      .passthrough()
      .optional(),
  })
  .passthrough();

export const MunitoResponse = z.object({ data: z.array(MunitoModel) }).passthrough();

export type MunitoModel = z.infer<typeof MunitoModel>;

type Modality = "text" | "audio" | "image" | "video" | "pdf";
type ReasoningOption = NonNullable<SyncedFullModel["reasoning_options"]>[number];

export const munito = {
  id: "munito",
  name: "Munito",
  modelsDir: "providers/munito/models",
  // The feed lists every model that the host sells, so a model that leaves the
  // feed leaves the catalog. parseModels refuses an empty feed.
  deleteMissing: true,
  // The leading comments name the feed and the wire path of each reasoning
  // control, so they follow the feed instead of the first file on disk.
  authoritativeHeaders: true,
  async fetchModels() {
    return fetchMunitoModels();
  },
  parseModels(raw) {
    return parseMunitoModels(raw);
  },
  translateModel(model, context) {
    const authored = context.authored(model.id);
    // An authored pointer wins, so a resolver miss never de-factors a committed file.
    const baseModel = authored?.base_model ?? resolveMunitoBaseModel(model.id);
    // Munito hosts other labs' models. A model with no lab file waits for a
    // [missing-model] issue instead of a full inline file.
    if (baseModel === undefined) return undefined;
    const built = buildMunitoModel(model, authored, baseModel);
    return { id: model.id, model: built, header: munitoHeader(model, built) };
  },
  sourceID(model) {
    return model.id;
  },
  missingModelID(model) {
    return model.id;
  },
  skippedNotice(ids) {
    if (ids.length === 0) return [];
    return [
      `${ids.length} Munito model(s) have no lab file to factor onto and were not created; add models/<lab>/<model>.toml, then re-sync: ${ids.map((id) => `\`${id}\``).join(", ")}`,
    ];
  },
} satisfies SyncProvider<MunitoModel>;

export async function fetchMunitoModels(fetcher: typeof fetch = fetch) {
  const response = await fetcher(API_ENDPOINT);
  if (!response.ok) {
    throw new Error(`Munito models request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// The feed repeats each model for each region (`qwen/qwen3.8-27b@eu`) and each
// service tier (`flex`, `priority`). This provider is the global host, so it
// keeps the standard-tier rows under the plain IDs.
export function parseMunitoModels(raw: unknown): MunitoModel[] {
  const rows = MunitoResponse.parse(raw).data.filter((model) =>
    model.deployment_region === "global" && model.service_tier === undefined && !model.id.includes("@")
  );
  // An empty 200 would otherwise delete every file, and the auto-merge allows up to 10 deletions.
  if (rows.length === 0) {
    throw new Error("Munito returned no standard-tier global models; refusing a destructive sync");
  }
  return rows;
}

export function resolveMunitoBaseModel(modelID: string): string | undefined {
  const exception = BASE_MODEL_EXCEPTIONS[modelID];
  if (exception !== undefined) return resolveModelMetadataBaseModel(exception);
  const [organization, ...model] = modelID.split("/");
  const prefix = organization === undefined ? undefined : LAB_PREFIXES[organization];
  return resolveModelMetadataBaseModel(prefix === undefined ? modelID : [prefix, ...model].join("/"));
}

export function buildMunitoModel(
  model: MunitoModel,
  authored: ExistingModel | undefined,
  baseModel: string,
): SyncedModel {
  const text = model.input_modalities.find((modality) => modality.type === "text");
  const output = textOutput(model);
  const parameters = output.supported_parameters;
  const input = modalities(model.input_modalities.map((modality) => modality.type));
  const limit = {
    context: text?.supported_inputs?.max_context_length?.value,
    output: output.max_length?.value,
  };
  const prices = [...text?.pricing ?? [], ...output.pricing ?? []];
  const prompt = perMillion(prices, "prompt");
  const completion = perMillion(prices, "completion");
  const cost = prompt !== undefined && completion !== undefined
    ? { input: prompt, output: completion, cache_read: perMillion(prices, "cached_prompt") }
    : authored?.cost;
  const reasoning = "reasoning" in parameters || "reasoning_effort" in parameters;
  return factorBaseModel(baseModel, {
    attachment: input.some((value) => value !== "text"),
    reasoning,
    reasoning_options: reasoning ? reasoningOptions(model) : undefined,
    tool_call: "tools" in parameters,
    structured_output: "structured_outputs" in parameters,
    temperature: "temperature" in parameters,
    // Munito passes the reasoning_content of an earlier assistant message to the
    // chat template, so interleaved thinking works on every reasoning model.
    interleaved: authored?.interleaved ?? (reasoning ? { field: "reasoning_content" } : undefined),
    status: authored?.status,
    cost,
    limit,
    modalities: { input, output: modalities(model.output_modalities.map((modality) => modality.type)) },
  }, limit, authored?.base_model_omit);
}

// AGENTS.md: an effort list with `none` already disables reasoning, so it takes
// no toggle. A reasoning model with no control in the feed waits for a human.
export function reasoningOptions(model: MunitoModel): ReasoningOption[] {
  const effort = effortValues(model);
  const options: ReasoningOption[] = [];
  if (thinkingSwitch(model) !== undefined && !effort.includes("none")) options.push({ type: "toggle" });
  if (effort.length > 0) options.push({ type: "effort", values: effort });
  if (options.length === 0) {
    throw new MissingReasoningOptionsError(
      model.id,
      `the Munito feed lists ${model.id} as a reasoning model but names no reasoning_effort levels or thinking switch`,
    );
  }
  return options;
}

function effortValues(model: MunitoModel): EffortValue[] {
  const values = textOutput(model).supported_parameters.reasoning_effort?.values ?? [];
  const unknown = values.filter((value) => !(EFFORT_VALUES as readonly string[]).includes(value));
  if (unknown.length > 0) {
    throw new MissingReasoningOptionsError(
      model.id,
      `the Munito feed lists reasoning_effort levels that the catalog schema does not know: ${unknown.join(", ")}`,
    );
  }
  return values as EffortValue[];
}

function thinkingSwitch(model: MunitoModel): string | undefined {
  const properties = model.passthrough_parameters?.chat_template_kwargs?.properties ?? {};
  return THINKING_SWITCHES.find((name) => properties[name]?.type === "boolean");
}

function textOutput(model: MunitoModel) {
  return model.output_modalities.find((modality) => modality.type === "text") ?? model.output_modalities[0]!;
}

// Per-token USD strings become per-1M, rounded to micro-dollars:
// 0.00000023 * 1e6 carries float error otherwise.
function perMillion(prices: z.infer<typeof Price>[], type: string): number | undefined {
  const entry = prices.find((price) => price.type === type && price.unit === "token");
  return entry === undefined ? undefined : Math.round(Number(entry.cost_usd) * 1e12) / 1e6;
}

const MODALITY_ORDER: Modality[] = ["text", "image", "video", "audio", "pdf"];

function modalities(values: string[]): Modality[] {
  const listed = new Set(values.map((value) => value.toLowerCase()).map((value) => value === "file" ? "pdf" : value));
  const result = MODALITY_ORDER.filter((value) => listed.has(value));
  return result.length > 0 ? result : ["text"];
}

// AGENTS.md wants the wire path of each reasoning control above the first key.
function munitoHeader(model: MunitoModel, built: SyncedModel): string {
  const lines = [`# Source: ${API_ENDPOINT} (public), standard tier on the global host.`];
  for (const option of built.reasoning_options ?? []) {
    if (option.type === "toggle") {
      lines.push(`# Toggle: chat_template_kwargs.${thinkingSwitch(model)} = true|false`);
    }
    if (option.type === "effort") {
      const off = option.values.includes("none") ? " (none disables reasoning, so there is no separate toggle)" : "";
      lines.push(`# Effort: reasoning_effort = ${option.values.join("|")}${off}`);
    }
  }
  return `${lines.join("\n")}\n`;
}
