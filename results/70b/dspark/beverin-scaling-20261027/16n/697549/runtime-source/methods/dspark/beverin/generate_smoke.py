"""Greedy generations from a running server, plus single-stream decode tok/s."""
import argparse
import time

import openai

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8001)
args = ap.parse_args()
c = openai.OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
m = c.models.list().data[0].id
for q in ["What is the capital of Switzerland? Answer in one sentence.",
          "Write a Python function that returns the n-th Fibonacci number."]:
    r = c.chat.completions.create(model=m, messages=[{"role": "user", "content": q}],
                                  temperature=0, max_tokens=200)
    print(f"Q: {q}\nA: {r.choices[0].message.content!r}\n", flush=True)
r = c.completions.create(model=m, prompt="The history of the Swiss Confederation begins",
                         temperature=0, max_tokens=64)
print("completion:", repr(r.choices[0].text), flush=True)
t = time.time()
r = c.completions.create(model=m, prompt="Once upon a time", temperature=0, max_tokens=512,
                         extra_body={"ignore_eos": True})
dt = time.time() - t
n = r.usage.completion_tokens
print(f"single-stream decode: {n} tokens in {dt:.1f} s = {n / dt:.1f} tok/s", flush=True)
