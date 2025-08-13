# LLM CPU Basics — Local Inference & Benchmarking

This project demonstrates running a Large Language Model (LLM) entirely on CPU using Hugging Face Transformers.

The script `run_llm_cpu.py` provides a **command-line interface** for:
- Sending prompts to any Hugging Face model (chat & non-chat)
- Switching between greedy decoding (deterministic) and probabilistic sampling
- Truncating long inputs to a maximum token limit
- Measuring generation performance and automatically exporting results to benchmarks.csv for model comparison

## 1. Requirements

```
pip install -r requirements.txt
```
## 2. Usage examples
### Greedy decoding (deterministic)
```
python run_llm_cpu.py --user "Explain overfitting briefly." --greedy
```
### Creative sampling
```
python run_llm_cpu.py --user "Give a funny analogy for neural networks." --temperature 0.9 --top_p 0.95
```
### Technical example with lower temperature
```
python run_llm_cpu.py --user "Write a Python function to compute F1 from precision and recall." --temperature 0.4
```
### Using a different model
```
python run_llm_cpu.py --model "microsoft/Phi-3-mini-4k-instruct" --user "Explain backpropagation simply." --greedy
```
### Non-chat model without template
```
python run_llm_cpu.py --model distilgpt2 --no_template --user "Q: What is overfitting in machine learning?\nA:" --temperature 0.7 --top_p 0.9 --top_k 50 --max_new_tokens 80
```
### With dynamic quantization
Dynamic quantization stores nn.Linear weights in 8-bit integers (qint8) to reduce memory usage and sometimes improve CPU inference speed.

```
python run_llm_cpu.py --user "Explain overfitting briefly." --greedy --quantize_dynamic
```

## 3. Parameters
```
| Parameter            | Description                                                                 | Default                                                                  |
|----------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `--model`            | Hugging Face model ID                                                       | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`                                     |
| `--system`           | System role instruction (ignored if `--no_template` is set)                 | `You are a concise assistant. Answer clearly and factually.`             |
| `--user`             | User prompt (required)                                                      | -                                                                        |
| `--temperature`      | Sampling temperature (higher = more creativity)                             | `0.5`                                                                    |
| `--top_p`            | Nucleus sampling probability mass                                           | `0.9`                                                                    |
| `--top_k`            | Maximum number of tokens considered at each step                            | `50`                                                                     |
| `--max_new_tokens`   | Maximum number of tokens to generate                                        | `160`                                                                    |
| `--greedy`           | Use greedy decoding (ignore sampling parameters)                            | disabled                                                                 |
| `--seed`             | RNG seed for reproducibility                                                | `None`                                                                   |
| `--no_template`      | Send raw prompt without chat formatting (for base models)                   | disabled                                                                 |
| `--quantize_dynamic` | Enable dynamic INT8 quantization for `nn.Linear` layers on CPU      	     | disabled									|                                            
```

## 4. Output
Each run prints two sections:
```
=== BENCHMARK === — model info, CPU details, generation mode, input/output token counts, speed, latency

=== REPLY === — the generated text
```
Example:
```
=== BENCHMARK ===
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
CPU: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel (4 cores / 8 threads)
Mode: GREEDY
Input tokens: 46
Generated tokens: 147
Time: 22.63 sec
Speed: 6.50 tok/s
Latency: 154.0 ms/token

=== REPLY ===
Overfitting is the phenomenon where a model trained on a small dataset (usually less than 80% of the total data) becomes incapable of generalizing well to new data that is not part of the training set. This happens because the model has learned patterns from the training data that are specific to the training data, which may not be applicable to other datasets. Overfitting can lead to poor performance on new data, as the model may struggle to make accurate predictions even when presented with new examples. To prevent overfitting, it's essential to use appropriate hyperparameters, regularization techniques, and data augmentation methods to ensure that the model is not overly sensitive to the training data.
```
## 5. CSV Benchmark Export
Every run automatically appends benchmark results to benchmarks.csv.
### CSV columns:
```
model,mode,input_tokens,generated_tokens,speed_tok_s,latency_ms_per_tok,cpu,cores,threads,temperature,top_p,top_k,greedy,dynamic_int8
```
### How to use it:
- Import into Excel or Pandas for analysis
- Compare models by size, decoding method, or parameters
- Sort by speed or latency to find the most efficient model for your CPU

## 6. Notes
- Chat models (e.g., Phi-3-mini, TinyLlama-Chat) use --system and --user to build a formatted prompt
- Base/non-chat models (e.g., GPT-2) require --no_template
- Performance heavily depends on CPU power and model size
- Running on CPU is significantly slower than GPU — benchmarks are useful for picking models
- `--quantize_dynamic` can significantly reduce RAM usage on CPU by storing `nn.Linear` weights in quantized INT8 format (`qint8`), with minimal accuracy loss for most models.




