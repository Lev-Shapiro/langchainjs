/**
 * Converts a raw OpenAI Chat Completions SSE chunk stream into LangChain
 * {@link ChatModelStreamEvent}s.
 *
 * @module
 */

import { OpenAI as OpenAIClient } from "openai";
import { finalizeContentBlock } from "@langchain/core/language_models/compat";
import type {
  ChatModelStreamEvent,
  FinishReason,
} from "@langchain/core/language_models/event";
import type { ContentBlock } from "@langchain/core/messages/content";
import type { UsageMetadata } from "@langchain/core/messages/metadata";

// ─── Public API ─────────────────────────────────────────────────

export interface ConvertOpenAICompletionsStreamOptions {
  streamUsage?: boolean;
  /** Provider id for passthrough events (default `"openai"`). */
  provider?: string;
  /** Optional per-chunk transform before conversion. */
  mapChunk?: (
    chunk: OpenAIClient.Chat.Completions.ChatCompletionChunk
  ) => OpenAIClient.Chat.Completions.ChatCompletionChunk;
}

type RawChunk = OpenAIClient.Chat.Completions.ChatCompletionChunk;
type BlockKey = "text" | "reasoning" | "audio" | `tool:${number}`;

/**
 * Convert an async iterable of raw OpenAI Chat Completions stream chunks into
 * LangChain `ChatModelStreamEvent`s with typed deltas.
 */
export async function* convertOpenAICompletionsStream(
  source: AsyncIterable<RawChunk>,
  options: ConvertOpenAICompletionsStreamOptions = {}
): AsyncGenerator<ChatModelStreamEvent> {
  const shouldStreamUsage = options.streamUsage ?? true;
  const provider = options.provider ?? "openai";
  const mapChunk = options.mapChunk;

  const blockAccumulators = new Map<
    number,
    // oxlint-disable-next-line @typescript-eslint/no-explicit-any
    Record<string, any>
  >();
  const blockKeyToIndex = new Map<BlockKey, number>();
  let nextBlockIndex = 0;
  let messageStarted = false;
  let usageSnapshot: UsageMetadata | undefined;
  let finishReason: FinishReason | undefined;
  let responseMetadata: Record<string, unknown> | undefined;
  let emittedProviderMetadata = false;

  const getOrCreateBlockIndex = (
    key: BlockKey,
    initial: Record<string, unknown>
  ): { index: number; isNew: boolean } => {
    const existing = blockKeyToIndex.get(key);
    if (existing !== undefined) {
      return { index: existing, isNew: false };
    }
    const index = nextBlockIndex++;
    blockKeyToIndex.set(key, index);
    blockAccumulators.set(index, { ...initial });
    return { index, isNew: true };
  };

  for await (let data of source) {
    if (mapChunk) {
      data = mapChunk(data);
    }
    if (!messageStarted) {
      messageStarted = true;
      yield {
        event: "message-start" as const,
        id: data.id,
      };
    }

    if (!emittedProviderMetadata && (data.model || data.service_tier)) {
      emittedProviderMetadata = true;
      yield {
        event: "provider" as const,
        provider,
        name: "stream_metadata",
        payload: {
          model: data.model,
          service_tier: data.service_tier,
        },
      };
    }

    if (data.usage && shouldStreamUsage) {
      usageSnapshot = buildUsageSnapshot(data.usage);
      yield { event: "usage" as const, usage: usageSnapshot };
    }

    const groqUsage = (data as { x_groq?: { usage?: GroqUsage } }).x_groq
      ?.usage;
    if (groqUsage && shouldStreamUsage) {
      usageSnapshot = buildGroqUsageSnapshot(groqUsage);
      yield { event: "usage" as const, usage: usageSnapshot };
    }

    const choice = data.choices?.[0];
    if (!choice) {
      continue;
    }

    if (choice.finish_reason != null) {
      finishReason = mapFinishReason(choice.finish_reason);
      responseMetadata = buildResponseMetadata(data, choice);
    }

    const { delta } = choice;
    if (!delta) {
      continue;
    }

    // ── Reasoning ──────────────────────────────────────────────
    const reasoningText = getReasoningDeltaText(delta);
    if (reasoningText) {
      const key: BlockKey = "reasoning";
      const { index, isNew } = getOrCreateBlockIndex(key, {
        type: "reasoning" as const,
        reasoning: "",
      });
      if (isNew) {
        yield {
          event: "content-block-start" as const,
          index,
          content: { type: "reasoning", reasoning: "" } as ContentBlock,
        };
      }
      const acc = blockAccumulators.get(index)!;
      acc.reasoning = (acc.reasoning ?? "") + reasoningText;
      yield {
        event: "content-block-delta" as const,
        index,
        delta: {
          type: "reasoning-delta" as const,
          reasoning: reasoningText,
        },
      };
    }

    // ── Text ───────────────────────────────────────────────────
    if (delta.content) {
      const key: BlockKey = "text";
      const { index, isNew } = getOrCreateBlockIndex(key, {
        type: "text" as const,
        text: "",
      });
      if (isNew) {
        yield {
          event: "content-block-start" as const,
          index,
          content: { type: "text", text: "" } as ContentBlock,
        };
      }
      const acc = blockAccumulators.get(index)!;
      acc.text = (acc.text ?? "") + delta.content;
      yield {
        event: "content-block-delta" as const,
        index,
        delta: { type: "text-delta" as const, text: delta.content },
      };
    }

    // ── Tool calls ─────────────────────────────────────────────
    if (Array.isArray(delta.tool_calls)) {
      for (const rawToolCall of delta.tool_calls) {
        const toolIndex = rawToolCall.index ?? 0;
        const key: BlockKey = `tool:${toolIndex}`;
        const { index, isNew } = getOrCreateBlockIndex(key, {
          type: "tool_call_chunk" as const,
          id: rawToolCall.id,
          name: rawToolCall.function?.name,
          args: "",
          index: toolIndex,
        });
        if (isNew) {
          yield {
            event: "content-block-start" as const,
            index,
            content: {
              type: "tool_call_chunk",
              id: rawToolCall.id,
              name: rawToolCall.function?.name,
              args: "",
              index: toolIndex,
            } as ContentBlock,
          };
        }

        const acc = blockAccumulators.get(index)!;
        if (rawToolCall.id != null) acc.id = rawToolCall.id;
        if (rawToolCall.function?.name != null) {
          acc.name = rawToolCall.function.name;
        }
        const argDelta = rawToolCall.function?.arguments ?? "";
        acc.args = (acc.args ?? "") + argDelta;
        yield {
          event: "content-block-delta" as const,
          index,
          delta: {
            type: "block-delta" as const,
            fields: {
              type: "tool_call_chunk",
              ...(acc.id != null ? { id: acc.id } : {}),
              ...(acc.name != null ? { name: acc.name } : {}),
              args: acc.args,
            },
          },
        };
      }
    }

    // ── Audio ──────────────────────────────────────────────────
    if (delta.audio) {
      const key: BlockKey = "audio";
      const { index, isNew } = getOrCreateBlockIndex(key, {
        type: "audio" as const,
        id: delta.audio.id,
        data: "",
        mimeType: "audio/pcm",
        transcript: delta.audio.transcript ?? "",
      });
      if (isNew) {
        yield {
          event: "content-block-start" as const,
          index,
          content: {
            type: "audio",
            id: delta.audio.id,
            data: "",
            mimeType: "audio/pcm",
            transcript: delta.audio.transcript ?? "",
          } as ContentBlock,
        };
      }
      const acc = blockAccumulators.get(index)!;
      if (delta.audio.transcript) {
        acc.transcript = (acc.transcript ?? "") + delta.audio.transcript;
        yield {
          event: "content-block-delta" as const,
          index,
          delta: {
            type: "block-delta" as const,
            fields: {
              type: "audio",
              transcript: acc.transcript,
            },
          },
        };
      }
      if (delta.audio.data) {
        acc.data = (acc.data ?? "") + delta.audio.data;
        yield {
          event: "content-block-delta" as const,
          index,
          delta: {
            type: "data-delta" as const,
            data: delta.audio.data,
            encoding: "base64" as const,
          },
        };
      }
    }

    // ── Legacy function_call ───────────────────────────────────
    if (delta.function_call) {
      yield {
        event: "provider" as const,
        provider,
        name: "function_call",
        payload: delta.function_call,
      };
    }

    // ── Logprobs passthrough ─────────────────────────────────────
    if (choice.logprobs) {
      yield {
        event: "provider" as const,
        provider,
        name: "logprobs",
        payload: choice.logprobs,
      };
    }
  }

  // Finalize all content blocks
  for (const [index, acc] of blockAccumulators) {
    const finalized = finalizeContentBlock(acc as ContentBlock);
    yield {
      event: "content-block-finish" as const,
      index,
      content: finalized,
    };
  }

  yield {
    event: "message-finish" as const,
    reason: finishReason,
    ...(usageSnapshot ? { usage: usageSnapshot } : {}),
    ...(responseMetadata ? { responseMetadata } : {}),
  };
}

// ─── Internal helpers ───────────────────────────────────────────

type GroqUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

function getReasoningDeltaText(
  // oxlint-disable-next-line @typescript-eslint/no-explicit-any
  delta: Record<string, any>
): string | undefined {
  const reasoning = delta.reasoning_content ?? delta.reasoning;
  return typeof reasoning === "string" && reasoning.length > 0
    ? reasoning
    : undefined;
}

function buildGroqUsageSnapshot(usage: GroqUsage): UsageMetadata {
  return {
    input_tokens: usage.prompt_tokens ?? 0,
    output_tokens: usage.completion_tokens ?? 0,
    total_tokens: usage.total_tokens ?? 0,
  };
}

function mapFinishReason(
  reason: OpenAIClient.Chat.Completions.ChatCompletionChunk.Choice["finish_reason"]
): FinishReason {
  switch (reason) {
    case "stop":
      return "stop";
    case "length":
    case "max_tokens":
      return "length";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "content_filter";
    default:
      return "stop";
  }
}

function buildUsageSnapshot(
  usage: OpenAIClient.Completions.CompletionUsage
): UsageMetadata {
  const inputTokenDetails = {
    ...(usage.prompt_tokens_details?.audio_tokens != null && {
      audio: usage.prompt_tokens_details.audio_tokens,
    }),
    ...(usage.prompt_tokens_details?.cached_tokens != null && {
      cache_read: usage.prompt_tokens_details.cached_tokens,
    }),
  };
  const outputTokenDetails = {
    ...(usage.completion_tokens_details?.audio_tokens != null && {
      audio: usage.completion_tokens_details.audio_tokens,
    }),
    ...(usage.completion_tokens_details?.reasoning_tokens != null && {
      reasoning: usage.completion_tokens_details.reasoning_tokens,
    }),
  };
  return {
    input_tokens: usage.prompt_tokens,
    output_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens,
    ...(Object.keys(inputTokenDetails).length > 0 && {
      input_token_details: inputTokenDetails,
    }),
    ...(Object.keys(outputTokenDetails).length > 0 && {
      output_token_details: outputTokenDetails,
    }),
  };
}

function buildResponseMetadata(
  data: RawChunk,
  choice: NonNullable<RawChunk["choices"]>[number]
): Record<string, unknown> {
  return {
    model_provider: "openai",
    model_name: data.model,
    system_fingerprint: data.system_fingerprint,
    service_tier: data.service_tier,
    finish_reason: choice.finish_reason,
    ...(data.usage ? { usage: data.usage } : {}),
  };
}
