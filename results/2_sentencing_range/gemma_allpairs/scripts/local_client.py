"""OpenAI-compatible client with round-robin across LOCAL_BASE_URLS (comma-sep)."""
import os, time, threading, itertools
from openai import OpenAI

_urls = os.environ.get("LOCAL_BASE_URLS", "").strip()
BASE_URLS = [u.strip() for u in _urls.split(",") if u.strip()] if _urls \
    else [os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1")]
BASE_URL = BASE_URLS[0]
MODEL = os.environ.get("LOCAL_MODEL", "gemma-4-31b")
API_KEY = os.environ.get("LOCAL_API_KEY", "local")
MAX_TOKENS = int(os.environ.get("LOCAL_MAX_TOKENS", "1024"))
TIMEOUT = float(os.environ.get("LOCAL_TIMEOUT", "600"))

_clients = [OpenAI(base_url=u, api_key=API_KEY, timeout=TIMEOUT) for u in BASE_URLS]
_rr = itertools.cycle(range(len(_clients)))
_lock = threading.Lock()

def _next():
    with _lock:
        return _clients[next(_rr)]

def info():
    return {"base_urls": BASE_URLS, "model": MODEL, "max_tokens": MAX_TOKENS, "timeout": TIMEOUT}

def call_local(system_prompt, user_prompt, max_retries=3, return_usage=False, max_tokens=None):
    payload = dict(model=MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.0, max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS)
    last_err = None
    for attempt in range(max_retries):
        try:
            comp = _next().chat.completions.create(**payload)
            text = (comp.choices[0].message.content or "").strip()
            if return_usage:
                u = getattr(comp, "usage", None)
                usage = {"prompt_tokens": getattr(u, "prompt_tokens", None),
                         "completion_tokens": getattr(u, "completion_tokens", None)} if u else {}
                return text, usage
            return text
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(5 * (2 ** attempt))
    raise RuntimeError(f"local model call failed after {max_retries} tries: {last_err}")
