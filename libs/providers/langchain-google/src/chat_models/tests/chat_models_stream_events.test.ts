import { describe, test, expect } from "vitest";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { ChatModelStream } from "@langchain/core/language_models/stream";
import type { BaseChatModelCallOptions } from "@langchain/core/language_models/chat_models";
import { ApiClient } from "../../clients/index.js";
import { ChatGoogle } from "../index.js";
import type { Gemini } from "../api-types.js";

class MockChunkStreamingResponse implements Response {
  readonly headers: Headers = new Headers();
  readonly ok = true;
  readonly redirected = false;
  readonly status = 200;
  readonly statusText = "OK";
  readonly type: ResponseType = "basic";
  readonly url = "http://localhost";
  readonly bodyUsed = false;
  readonly body: ReadableStream<Uint8Array<ArrayBuffer>>;

  constructor(chunks: Gemini.GenerateContentResponse[]) {
    const encoder = new TextEncoder();
    this.body = new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
          );
        }
        controller.close();
      },
    });
  }

  async arrayBuffer(): Promise<ArrayBuffer> {
    throw new Error("Not implemented");
  }
  async blob(): Promise<Blob> {
    throw new Error("Not implemented");
  }
  async formData(): Promise<FormData> {
    throw new Error("Not implemented");
  }
  async json(): Promise<unknown> {
    throw new Error("Not implemented");
  }
  async text(): Promise<string> {
    throw new Error("Not implemented");
  }
  async bytes(): Promise<Uint8Array<ArrayBuffer>> {
    throw new Error("Not implemented");
  }
  clone(): Response {
    throw new Error("Not implemented");
  }
}

class MockStreamingApiClient extends ApiClient {
  constructor(private readonly chunks: Gemini.GenerateContentResponse[]) {
    super();
  }

  async fetch(): Promise<Response> {
    return new MockChunkStreamingResponse(this.chunks);
  }

  hasApiKey(): boolean {
    return true;
  }
}

const streamChunks: Gemini.GenerateContentResponse[] = [
  {
    candidates: [{ content: { parts: [{ text: "Hello" }] } }],
  },
  {
    candidates: [{ content: { parts: [{ text: " world" }] } }],
  },
];

describe("ChatGoogle._streamChatModelEvents", () => {
  test("ChatModelStream.text end-to-end", async () => {
    const model = new ChatGoogle({
      model: "gemini-2.0-flash",
      apiKey: "fake-key",
      apiClient: new MockStreamingApiClient(streamChunks),
    });
    const stream = new ChatModelStream(
      model._streamChatModelEvents([], {} as BaseChatModelCallOptions)
    );
    const text = await stream.text;
    expect(text).toContain("Hello");
    expect(text).toContain("world");
  });

  test("emits message lifecycle", async () => {
    const model = new ChatGoogle({
      model: "gemini-2.0-flash",
      apiKey: "fake-key",
      apiClient: new MockStreamingApiClient(streamChunks),
    });
    const events: ChatModelStreamEvent[] = [];
    for await (const event of model._streamChatModelEvents(
      [],
      {} as BaseChatModelCallOptions
    )) {
      events.push(event);
    }
    expect(events.map((e) => e.event)).toContain("message-start");
    expect(events.map((e) => e.event)).toContain("message-finish");
  });
});
