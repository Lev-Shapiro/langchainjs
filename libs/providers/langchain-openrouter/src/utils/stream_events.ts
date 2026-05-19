/**
 * Converts OpenRouter SSE stream chunks into LangChain ChatModelStreamEvents.
 *
 * @module
 */

import { OpenAI as OpenAIClient } from "openai";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { convertOpenAICompletionsStream } from "@langchain/openai";
import type { StreamingChunkData } from "../converters/messages.js";

export interface ConvertOpenRouterStreamOptions {
  streamUsage?: boolean;
}

function mapOpenRouterChunkToOpenAI(
  data: StreamingChunkData
): OpenAIClient.Chat.Completions.ChatCompletionChunk {
  const choice = data.choices?.[0];
  if (
    choice?.delta &&
    typeof choice.delta.reasoning === "string" &&
    choice.delta.reasoning_content == null
  ) {
    return {
      ...(data as unknown as OpenAIClient.Chat.Completions.ChatCompletionChunk),
      choices: [
        {
          ...choice,
          delta: {
            ...choice.delta,
            reasoning_content: choice.delta.reasoning,
          },
        },
      ],
    } as OpenAIClient.Chat.Completions.ChatCompletionChunk;
  }
  return data as unknown as OpenAIClient.Chat.Completions.ChatCompletionChunk;
}

export async function* convertOpenRouterStream(
  source: AsyncIterable<StreamingChunkData>,
  options: ConvertOpenRouterStreamOptions = {}
): AsyncGenerator<ChatModelStreamEvent> {
  async function* mapped() {
    for await (const chunk of source) {
      yield mapOpenRouterChunkToOpenAI(chunk);
    }
  }
  yield* convertOpenAICompletionsStream(mapped(), {
    ...options,
    provider: "openrouter",
  });
}
