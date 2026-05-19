import type { LangChainMatchers } from "./matchers.js";

declare module "@vitest/expect" {
  // oxlint-disable-next-line @typescript-eslint/no-explicit-any
  interface Assertion<T = any> extends LangChainMatchers<Assertion<T>> {}
  // oxlint-disable-next-line @typescript-eslint/no-explicit-any
  interface PromisifyAssertion<T = any> extends LangChainMatchers<
    PromisifyAssertion<T>
  > {}
}

export type { LangChainMatchers };
