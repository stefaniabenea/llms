import argparse, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import platform
import psutil
from pathlib import Path
import csv
try:
    from torch.ao.quantization import quantize_dynamic as _quantize_dynamic
except Exception:
    from torch.quantization import quantize_dynamic as _quantize_dynamic

def load(model_name: str, use_dynamic_quant: bool = False):
    
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    mdl = maybe_dynamic_quantize(mdl, use_dynamic_quant)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    mdl.eval()
    return tok, mdl

def build_prompt(tok, system_msg: str, user_msg: str) -> str:
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(tok, mdl, prompt: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int, greedy: bool, max_input_tokens: int = 3072):
    
    inputs = tok(prompt, return_tensors="pt")
    if inputs["input_ids"].shape[1] > max_input_tokens:
        for k in ("input_ids", "attention_mask"):
            inputs[k] = inputs[k][:, -max_input_tokens:]

    kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=1.1
    )
    if greedy:
        # Greedy decoding: always pick the highest probability token
        kwargs.update(dict(do_sample=False))
    else:
        # Sampling: probabilistic decoding with temperature, top_p, and top_k
        kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k))

    t0 = time.perf_counter()
    with torch.no_grad():
        out = mdl.generate(**inputs, **kwargs)
    dt = time.perf_counter() - t0

    input_len = inputs["input_ids"].shape[1]
    new_tokens = out.shape[-1] - input_len
    reply = tok.decode(out[0, input_len:], skip_special_tokens=True).strip()
    tps = new_tokens / dt if dt > 0 else float("inf")
    ms_per_tok = (dt / max(new_tokens, 1)) * 1000
    return reply, new_tokens, dt, tps, ms_per_tok, input_len

def print_benchmark_info(model_name, temperature, top_p, top_k, greedy,
                         input_len, new_tokens, dt, tps, ms_per_tok):
    cpu_info = platform.processor() or "Unknown CPU"
    cores = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True)
    mode = "GREEDY" if greedy else "SAMPLING"

    print("\n=== BENCHMARK ===")
    print(f"Model: {model_name}")
    print(f"CPU: {cpu_info} ({cores} cores / {threads} threads)")
    print(f"Mode: {mode}")
    if mode == "SAMPLING":
        print(f"Decoding params: temperature={temperature}, top_p={top_p}, top_k={top_k}")
    print(f"Input tokens: {input_len}")
    print(f"Generated tokens: {new_tokens}")
    print(f"Time: {dt:.2f} sec")
    print(f"Speed: {tps:.2f} tok/s")
    print(f"Latency: {ms_per_tok:.1f} ms/token")
    return cpu_info, cores, threads, mode

def write_benchmark_csv(model_name, cpu_info, cores, threads, mode,
                        greedy, temperature, top_p, top_k,
                        input_len, new_tokens, tps, ms_per_tok,
                        path="benchmarks.csv", dynamic_int8=0):
    
    p = Path(path)
    write_header = not p.exists()
    header = [
        "model","mode","input_tokens","generated_tokens","speed_tok_s","latency_ms_per_tok",
        "cpu","cores","threads","temperature","top_p","top_k","greedy","dynamic_int8"
    ]
    row = [
        model_name, mode, input_len, new_tokens, round(tps, 2), round(ms_per_tok, 1),
        cpu_info, cores, threads, temperature, top_p, top_k, int(bool(greedy)), int(bool(dynamic_int8))
    ]
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)  
        if write_header:
            w.writerow(header)
        w.writerow(row)

def maybe_dynamic_quantize(model, enable: bool):
    if not enable:
        return model
    
    return _quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face model ID to load.")
    ap.add_argument("--system", default="You are a concise assistant. Answer clearly and factually.", help="System role instruction.")
    ap.add_argument("--user", required=True, help="User message (the question or prompt).")
    ap.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (higher = more random).")
    ap.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling probability mass.")
    ap.add_argument("--top_k", type=int, default=50, help="Top-k sampling limit.")
    ap.add_argument("--max_new_tokens", type=int, default=160, help="Maximum number of tokens to generate.")
    ap.add_argument("--greedy", action="store_true", help="Use greedy decoding (ignore sampling parameters).")
    ap.add_argument("--seed", type=int, default=None, help="Set RNG seed for reproducible sampling.")
    ap.add_argument("--no_template", action="store_true", help="Send raw prompt without chat template (for base models).")
    ap.add_argument("--quantize_dynamic", action="store_true", help="Activate dynamic quantization INT8 on CPU (nn.Linear).")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    tok, mdl = load(args.model, use_dynamic_quant=args.quantize_dynamic)
    prompt = args.user if args.no_template else build_prompt(tok, args.system, args.user)
    reply, new_tokens, dt, tps, ms_per_tok, input_len = generate(tok, mdl, prompt, args.temperature, args.top_p, args.top_k, args.max_new_tokens, args.greedy)
    
    cpu_info, cores, threads, mode = print_benchmark_info(
    args.model,
    args.temperature,
    args.top_p,
    args.top_k,
    args.greedy,
    input_len,
    new_tokens,
    dt,
    tps,
    ms_per_tok
    )
    write_benchmark_csv(
    args.model, cpu_info, cores, threads, mode,
    args.greedy, args.temperature, args.top_p, args.top_k,
    input_len, new_tokens, tps, ms_per_tok, path="benchmarks.csv", dynamic_int8=args.quantize_dynamic
)

    print("\n=== REPLY ===")
    print(reply)
    

if __name__ == "__main__":
    main()
