---
"@langchain/core": patch
"@langchain/openai": patch
"@langchain/ollama": patch
"@langchain/aws": patch
"@langchain/cohere": patch
"@langchain/cloudflare": patch
"@langchain/mistralai": patch
"@langchain/google": patch
"@langchain/google-genai": patch
"@langchain/google-common": patch
"@langchain/groq": patch
"@langchain/openrouter": patch
"@langchain/deepseek": patch
"@langchain/together-ai": patch
"@langchain/fireworks": patch
"@langchain/xai": patch
"@langchain/perplexity": patch
"@langchain/ibm": patch
---

feat(providers): native `streamV2` ChatModelStreamEvent protocol across chat providers

Add provider-specific `convert*Stream` utilities and `_streamChatModelEvents`
overrides so `model.streamV2()` exposes typed sub-streams (`.text`, `.reasoning`,
`.toolCalls`, `.usage`, `.output`). OpenAI Completions conversion lives in
`@langchain/core/language_models/openai_completions_stream` and is re-exported
from `@langchain/openai`; OpenAI-compatible providers import from core directly.
Legacy `_streamResponseChunks` paths are unchanged.
