#!/usr/bin/env python3
"""
Centralized model configuration.
Add new models here and they will be automatically included in all experiments.
"""

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

# Available models for similarity prediction experiments
# Format: "model_id": {"display_name": "Display Name", "backend": "backend_type"}
MODELS = {
    "gpt4": {
        "display_name": "GPT-4",
        "backend": "openai",
        "model_name": "gpt-4.1"
    },
    "gpt5mini": {
        "display_name": "GPT-5 Mini",
        "backend": "openai",
        "model_name": "gpt-5-mini-2025-08-07"
    },
    "dicta": {
        "display_name": "DictaLM",
        "backend": "dicta",
        "model_name": "dicta-il/DictaLM-3.0-24B-Thinking:publicai"
    },
    "mistral": {
        "display_name": "Mistral",
        "backend": "nim",
        "model_name": "mistralai/mistral-small-24b-instruct"
    },
    "qwen": {
        "display_name": "Qwen",
        "backend": "nim",
        "model_name": "qwen/qwen2.5-7b-instruct"
    },
    "llama": {
        "display_name": "Llama",
        "backend": "nim",
        "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1"
    },
    "nemotron": {
        "display_name": "Nemotron",
        "backend": "nim",
        "model_name": "nvidia/nvidia-nemotron-nano-9b-v2"
    },
    "gpt_oss": {
        "display_name": "GPT-OSS",
        "backend": "nim",
        "model_name": "openai/gpt-oss-120b"
    },
    "random": {
        "display_name": "Random Baseline",
        "backend": "random",
        "model_name": None
    },
    # New models added and tested
    "nemotron3_nano": {
        "display_name": "Nemotron-3-Nano-30B",
        "backend": "nim",
        "model_name": "nvidia/nemotron-3-nano-30b-a3b"
    },
    # Routed to Hugging Face Router in similarity_experiment.call_nim (NIM 410); set NIM_FORCE_QWEN3_235B=1 to use NIM.
    "qwen3_235b": {
        "display_name": "Qwen3-235B",
        "backend": "nim",
        "model_name": "qwen/qwen3-235b-a22b"
    },
    "llama3_70b": {
        "display_name": "Llama3-70B-Instruct",
        "backend": "nim",
        "model_name": "meta/llama-3.1-70b-instruct"
    },
    "gpt52": {
        "display_name": "GPT-5.2",
        "backend": "openai",
        "model_name": "gpt-5.2"
    },
    "gemma3_27b": {
        "display_name": "Gemma-3-27B",
        "backend": "hf_router",
        "model_name": "google/gemma-3-27b-it:scaleway",
    },
    "gpt51_thinking": {
        "display_name": "GPT-5.1 (with reasoning)",
        "backend": "openai",
        "model_name": "gpt-5.1",
        "use_reasoning": True
    },
}

# Default models to run (can be overridden)
DEFAULT_MODELS = ["gpt4", "gpt5mini", "dicta", "mistral"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_list(models=None):
    """Get list of model IDs to run.
    
    Args:
        models: List of model IDs, or None to use DEFAULT_MODELS
        
    Returns:
        List of model IDs
    """
    if models is None:
        return DEFAULT_MODELS.copy()
    
    # Validate models exist
    invalid = [m for m in models if m not in MODELS]
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Available: {list(MODELS.keys())}")
    
    return models


def get_model_info(model_id):
    """Get model information.
    
    Args:
        model_id: Model ID string
        
    Returns:
        Dict with model info, or None if not found
    """
    return MODELS.get(model_id)


def list_available_models():
    """List all available models."""
    return list(MODELS.keys())


def add_model(model_id, display_name, backend, model_name=None):
    """Add a new model to the configuration.
    
    Args:
        model_id: Unique identifier (e.g., "new_model")
        display_name: Human-readable name
        backend: "openai", "nim", "dicta", or "random"
        model_name: Model identifier for the backend (optional for random)
    """
    if model_id in MODELS:
        raise ValueError(f"Model {model_id} already exists!")
    
    MODELS[model_id] = {
        "display_name": display_name,
        "backend": backend,
        "model_name": model_name
    }
    
    print(f"✅ Added model: {model_id} ({display_name})")


if __name__ == "__main__":
    print("Available Models:")
    print("=" * 60)
    for model_id, info in MODELS.items():
        print(f"{model_id:15} | {info['display_name']:20} | {info['backend']:10}")
    print(f"\nDefault models: {', '.join(DEFAULT_MODELS)}")

