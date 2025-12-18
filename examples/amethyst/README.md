# Amethyst - Intellect-3 RL Training

In this example, we demonstrate how to train `Refined-Gem-4B-Thinking` on the Intellect-3 RL environments. This example highlights several key features of PRIME-RL:

- **Single-file configuration**: All training settings (trainer, orchestrator, and inference) are specified in a single `rl.toml` file
- **LoRA training**: Efficient fine-tuning using LoRA (Low-Rank Adaptation) on attention and MLP layers
- **Multi-environment training**: Trains across four Intellect-3 domains (code, logic, math, science)
- **Tool calling support**: Configured for Hermes-style tool calling
- **Online difficulty buffer**: Uses difficulty-based sampling to ensure rollouts have strictly non-zero advantages

> This example runs on 4 GPUs (3 for inference, 1 for training).

## Setup

Install the Intellect-3 environments:

```bash
prime env install primeintellect/i3-code
prime env install primeintellect/i3-logic
prime env install primeintellect/i3-math
prime env install primeintellect/i3-science
```

Verify installation:

```bash
uv run python -c "import i3_code; import i3_logic; import i3_math; import i3_science"
```

Start the tmux session:

```bash
bash scripts/tmux.sh
```

## Task

The Intellect-3 environments test the model across four reasoning domains:

1. **i3-code**: Programming and code reasoning tasks
2. **i3-logic**: Logical reasoning and deduction problems
3. **i3-math**: Mathematical problem solving
4. **i3-science**: Scientific reasoning and knowledge

Each environment provides domain-specific challenges to develop robust reasoning capabilities across multiple areas.

## Scoring

The Intellect-3 environments use domain-specific scoring rubrics tailored to each task type. Each environment evaluates correctness based on the expected output format and solution quality for its respective domain.

## Configuration

This example uses a **single `rl.toml` file** that contains all configuration for trainer, orchestrator, and inference in a single place. This simplifies configuration for single-node training via `rl.py`. 

Key configuration highlights:

- **LoRA training**: Rank 64, alpha 64 for efficient fine-tuning
- **Tool calling**: Uses Hermes parser for automatic tool selection with Refined-Gem-4B-Thinking
- **Multi-environment**: Trains across all four Intellect-3 environments simultaneously
- **Online difficulty buffer**: Uses difficulty-based sampling with 2x oversampling

## Baseline Evaluation

Start the inference server:

```bash
# In the `Inference` pane
uv run inference --model.name qingy2024/Refined-Gem-4B-Thinking --model.enable_auto_tool_choice --model.tool_call_parser hermes
```

Evaluate the base model on each environment:

```bash
# In the `Trainer` pane
uv run vf-eval i3-code -m qingy2024/Refined-Gem-4B-Thinking -b http://localhost:8000/v1 -n 20 --max-tokens 2048
uv run vf-eval i3-logic -m qingy2024/Refined-Gem-4B-Thinking -b http://localhost:8000/v1 -n 20 --max-tokens 2048
uv run vf-eval i3-math -m qingy2024/Refined-Gem-4B-Thinking -b http://localhost:8000/v1 -n 20 --max-tokens 2048
uv run vf-eval i3-science -m qingy2024/Refined-Gem-4B-Thinking -b http://localhost:8000/v1 -n 20 --max-tokens 2048
```

## RL Training

Train with the unified config file:

```bash
# In the `Trainer` pane
uv run rl @ examples/amethyst/rl.toml \
  --wandb.project your-project-name \
  --wandb.name your-run-name
```

The unified config file automatically configures:
- **Trainer**: LoRA fine-tuning with specified hyperparameters
- **Orchestrator**: Rollout generation across all four Intellect-3 environments with tool calling enabled
- **Inference**: vLLM server for Refined-Gem-4B-Thinking with tool parsing enabled

This will write weight checkpoints in `outputs/weights/step_*`. Upload the final checkpoint to HuggingFace:

```bash
uv run hf upload <user>/Amethyst-4B-RL outputs/weights/step_1000
```

## Evaluation

Evaluate your trained model:

```bash
# In the `Inference` pane
uv run inference --model.name <user>/Amethyst-4B-RL --inference.model.enable_auto_tool_choice true --inference.model.tool_call_parser hermes
```

```bash
# In the `Trainer` pane
uv run vf-eval i3-code -m <user>/Amethyst-4B-RL -b http://localhost:8000/v1 -n 20 --max-tokens 2048
uv run vf-eval i3-logic -m <user>/Amethyst-4B-RL -b http://localhost:8000/v1 -n 20 --max-tokens 2048
uv run vf-eval i3-math -m <user>/Amethyst-4B-RL -b http://localhost:8000/v1 -n 20 --max-tokens 2048
uv run vf-eval i3-science -m <user>/Amethyst-4B-RL -b http://localhost:8000/v1 -n 20 --max-tokens 2048
```

## Environment Configuration

You can configure environment-specific arguments in your `rl.toml`:

```toml
[[orchestrator.env]]
id = "primeintellect/i3-code"
args = { max_turns = 5 }

[[orchestrator.env]]
id = "primeintellect/i3-logic"
args = { max_turns = 5 }

[[orchestrator.env]]
id = "primeintellect/i3-math"
args = { max_turns = 5 }

[[orchestrator.env]]
id = "primeintellect/i3-science"
args = { max_turns = 5 }
```

## Notes

- Tool calling requires `enable_auto_tool_choice = true` and a compatible parser (Hermes is recommended)
- The model is trained across all four Intellect-3 environments to develop robust multi-domain reasoning
- Adjust `batch_size` and `rollouts_per_example` in the config based on your GPU memory