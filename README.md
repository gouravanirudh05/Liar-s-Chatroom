# The Liar's Chatroom: methodology

Final submission: `submission_v5_full.csv`, public 0.99555, private 0.99269.

## Models
| Model | Parameters | Licence | Role |
|---|---|---|---|
| [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) (Qwen3-14B, 4-bit AWQ) | 14.8B | Apache-2.0 | main reasoner (thinking mode) |
| [Qwen/Qwen3-32B-AWQ](https://huggingface.co/Qwen/Qwen3-32B-AWQ) (Qwen3-32B, 4-bit AWQ) | 32.8B | Apache-2.0 | arbiter on contested cases |

All inference is local with vLLM on Kaggle GPU T4 x2. No closed or hosted model APIs, no fine-tuning, no external data.

## Pipeline
1. **Exact rule modules** (`liars/rules.py`). There are 27 small parsers, one per evidence family: time windows, vehicle round trips, ages, rotas, tank levels, bookings, ownership chains and others. Each one flags only a message it can prove contradicts the evidence. A case gets a label only when every flag points to one sender. On train these modules cover 57% of cases at 100% precision (`python3 scripts/eval_rules.py`). When the evidence contains unfamiliar qualifiers (except, unless, if, or, ...), the rules abstain and the LLM decides, which protects against scenario families not seen in train.
2. **LLM reasoning** (`liars/prompt.py`, `kaggle/llm_baseline/infer.py`). Qwen3-14B runs in thinking mode on the cases the rules don't cover. The prompt states the competition's contradiction rules, drops questions and small talk, labels each message with the sender's name, and asks for the number of the contradicting message, which is then mapped to its sender.
3. **Hard-case refinement** (`kaggle/refine/main.py`). Cases whose reasoning was long or truncated get 2 more samples with a 10k-token budget, plus budget forcing for truncated samples. The final label is a 2-of-3 majority that includes the first answer.
4. **Second opinions** (`kaggle/verify/main.py`). Every single-sample LLM case and every rule-covered case gets one more independent sample. Disagreements get 3 more samples, and contested cases get 6 more. A rule is overridden only when at least 80% of 10 LLM votes agree on another sender. That threshold was set on train, where 7 of 10 votes against a correct rule still occurred.
5. **32B arbiter** (`kaggle/verify/main.py`, `liars-32b-*` builds). Qwen3-32B casts 3 votes on the 49 still-contested test cases. Its label is used only when all 3 votes agree (and for rule cases, only when at least half of the 14B votes agree too). Validation on train: 32B scored 96/96 on hard uncovered cases where 14B scored 94/96, and the policy decided all 10 train counterpart cases correctly.

The `scratch/combine_v*.py` scripts implement each decision policy offline from the saved votes.

## Validation (train only)
Every policy was tested on train before use. Test rows were never printed, read or hand-labelled, and no rule was designed from test.csv; test data only flowed through the pipeline. An AI coding assistant helped write code; it did not label any data.

| Version | Public | Private |
|---|---|---|
| v1 rules + Qwen3-14B | 0.98777 | n/a |
| v2b hard-case refinement | 0.99222 | 0.99038 |
| v4 second opinions + rule overrides | 0.99444 | 0.99192 |
| **v5 + Qwen3-32B arbiter** | **0.99555** | **0.99269** |

## Reproduce
`bash setup.sh`, then build and push the kernels in order (each one reads the previous kernel's outputs):
`scripts/build_kernel.py --mode test` → `scripts/build_refine.py '{"k":2,"max_tokens":10000,"max_model_len":12288}'` → `scripts/build_verify.py` (verify-llm, verify-rule, vote-extra*, 32b-big*) → `scratch/combine_v5.py`.
Push each with `kaggle kernels push -p kaggle/build/<slug> --accelerator NvidiaTeslaT4`. Competition data is not redistributed here.
