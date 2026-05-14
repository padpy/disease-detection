"""OpenAI-compatible chatbot service that wraps the BLIP_Qwen vision-LLM
checkpoint produced by ``LLM_work/how_to_use/assemble_model_demo.py``.

The model is multimodal (image + prompt). For ``/v1/chat/completions`` we treat
the most recent ``image_url`` content part as the active image and the
remaining text content as the prompt. Loading is lazy because constructing the
model takes seconds and pulls weights off disk.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image


DEFAULT_MODEL_NAME = "gopher-eye-grape-leaf"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1
DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


class ChatbotError(Exception):
    """Maps cleanly onto an OpenAI-style error response."""

    def __init__(self, message: str, status_code: int = 400, error_type: str = "invalid_request_error",
                 param: Optional[str] = None, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_payload(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class ChatbotService:
    """Lazily loads the BLIP_Qwen model and answers OpenAI-style chat requests."""

    def __init__(
        self,
        ckpt_dir: str,
        llm_src_root: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        self.llm_src_root = Path(llm_src_root).expanduser().resolve() if llm_src_root else None
        self.model_name = model_name
        self._device_override = device

        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[str] = None
        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._device = None

    # ---------------------- model loading ----------------------

    def preload(self):
        """Eagerly load weights now so the first chat request doesn't pay the
        load cost. Re-raises ChatbotError on failure with the same envelope
        chat_completion would return; the failure is also cached in
        _load_error so subsequent requests fail fast without re-trying."""
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._load_error is not None:
                raise ChatbotError(self._load_error, status_code=503, error_type="server_error")
            try:
                self._load()
                self._loaded = True
            except ChatbotError:
                raise
            except Exception as exc:
                self._load_error = f"failed to load chatbot model: {exc}"
                raise ChatbotError(self._load_error, status_code=503, error_type="server_error") from exc

    def _load(self):
        if not self.ckpt_dir.is_dir():
            raise ChatbotError(
                f"chatbot checkpoint directory not found: {self.ckpt_dir}",
                status_code=503,
                error_type="server_error",
            )

        if self.llm_src_root is not None and str(self.llm_src_root) not in sys.path:
            sys.path.insert(0, str(self.llm_src_root))

        try:
            import torch  # noqa: WPS433 - imported lazily so the rest of the
            from peft import PeftModel  # noqa: WPS433  server can boot without
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433  these heavy deps installed.

            from src.BLIP_Qwen.BLIP import BLIP2Model
            from src.BLIP_Qwen.blip2_support import build_blip2_image_processors
            from src.BLIP_Qwen.cross_model.projector import MLPProjector
            from src.BLIP_Qwen.model import QwenWithBLIPPrefix
        except ImportError as exc:
            raise ChatbotError(
                f"chatbot dependencies not installed: {exc}. "
                "Install transformers, peft, torch and ensure GOPHER_EYE_LLM_DIR points at LLM_work/.",
                status_code=503,
                error_type="server_error",
            ) from exc

        self._torch = torch

        if self._device_override:
            device = torch.device(self._device_override)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        load_dtype = torch.float16 if device.type == "cuda" else torch.float32

        meta = self._read_meta(self.ckpt_dir)
        base_model_id = meta["base_model"]
        blip2_model_id = meta["blip2_model"]
        meta_dict = meta["meta"]

        projector_path = self.ckpt_dir / "projector.pt"
        if not projector_path.exists():
            raise ChatbotError(
                f"missing projector checkpoint: {projector_path}",
                status_code=503,
                error_type="server_error",
            )

        qformer_stage_dir = self._resolve_qformer_stage_dir(self.ckpt_dir, meta_dict)

        qwen = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=load_dtype, trust_remote_code=True,
        ).to(device)
        qwen.eval()
        qwen = PeftModel.from_pretrained(qwen, str(self.ckpt_dir)).merge_and_unload().to(device)
        qwen.eval()

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        _, image_processor, resolved_lavis_model_type = build_blip2_image_processors(
            blip2_model_id=blip2_model_id,
            lavis_model_type=meta_dict.get("lavis_model_type"),
        )

        blip = BLIP2Model(
            blip2_model_id=blip2_model_id,
            device=str(device),
            dtype=load_dtype,
            qformer_stage1_dir=str(qformer_stage_dir),
            num_query_token=meta_dict.get("num_query_token", meta_dict.get("prefix_len", 32)),
            cross_attention_freq=meta_dict.get("cross_attention_freq", 2),
            lavis_model_type=resolved_lavis_model_type,
            freeze_vision=True,
            freeze_qformer=True,
            train_query_tokens=meta_dict.get("train_query_tokens", False),
        )
        blip.eval()

        projector = MLPProjector(
            in_dim=blip.qformer_dim,
            out_dim=qwen.config.hidden_size,
            hidden_dim=2 * qwen.config.hidden_size,
            use_residual=True,
            dropout=0.0,
        ).to(device, dtype=qwen.get_input_embeddings().weight.dtype)
        projector.load_state_dict(torch.load(projector_path, map_location=device), strict=True)
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False

        self._model = QwenWithBLIPPrefix(qwen=qwen, blip=blip, projector=projector)
        self._model.eval()
        self._tokenizer = tokenizer
        self._image_processor = image_processor

    @staticmethod
    def _read_meta(ckpt_dir: Path) -> Dict[str, Any]:
        meta_path = ckpt_dir / "model_meta.json"
        blip_txt_path = ckpt_dir / "blip2model.txt"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if blip_txt_path.exists():
            blip2_model_id = blip_txt_path.read_text(encoding="utf-8").strip()
        else:
            blip2_model_id = meta.get("blip2_model", "Salesforce/blip2-opt-2.7b")
        return {
            "base_model": meta.get("base_model", "Qwen/Qwen3-1.7B"),
            "blip2_model": blip2_model_id,
            "meta": meta,
        }

    @staticmethod
    def _resolve_qformer_stage_dir(ckpt_dir: Path, meta: Dict[str, Any]) -> Path:
        if (ckpt_dir / "stage1_meta.json").exists():
            return ckpt_dir
        raw = meta.get("qformer_stage1_dir")
        if raw:
            for candidate in (Path(raw), ckpt_dir.parent / raw):
                resolved = candidate.expanduser().resolve()
                if resolved.is_dir():
                    return resolved
        raise ChatbotError(
            "could not locate Qformer stage1 directory",
            status_code=503,
            error_type="server_error",
        )

    # ---------------------- request handling ----------------------

    def list_models(self) -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "gopher-eye",
                }
            ],
        }

    def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ChatbotError("'messages' must be a non-empty list", param="messages")

        prompt, image = self._messages_to_prompt_and_image(messages)
        if image is None:
            raise ChatbotError(
                "this model is multimodal — at least one user message must include an image_url content part",
                param="messages",
            )

        max_new_tokens = self._coerce_max_tokens(payload)
        temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
        do_sample = temperature > 0 and bool(payload.get("do_sample", temperature != 0))

        text = self._generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode(text)) if text else 0
        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @staticmethod
    def _coerce_max_tokens(payload: Dict[str, Any]) -> int:
        # OpenAI now prefers ``max_completion_tokens`` but older clients still send ``max_tokens``.
        for key in ("max_completion_tokens", "max_tokens"):
            value = payload.get(key)
            if value is None:
                continue
            try:
                ivalue = int(value)
            except (TypeError, ValueError) as exc:
                raise ChatbotError(f"'{key}' must be an integer", param=key) from exc
            if ivalue <= 0:
                raise ChatbotError(f"'{key}' must be > 0", param=key)
            return ivalue
        return DEFAULT_MAX_NEW_TOKENS

    def _messages_to_prompt_and_image(self, messages: List[Dict[str, Any]]) -> Tuple[str, Optional[Image.Image]]:
        rendered_lines: List[str] = []
        latest_image: Optional[Image.Image] = None
        last_user_text: Optional[str] = None

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ChatbotError(f"messages[{idx}] must be an object", param="messages")
            role = msg.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ChatbotError(f"messages[{idx}].role must be one of system|user|assistant", param="messages")
            text_part, image_part = self._extract_content(msg.get("content"), idx)
            if image_part is not None:
                latest_image = image_part
            if text_part:
                rendered_lines.append(f"{role.capitalize()}: {text_part}")
                if role == "user":
                    last_user_text = text_part

        # If the conversation has prior turns, we want the model to continue
        # the assistant's reply. Append an "Assistant:" cue so generation
        # produces a natural continuation rather than echoing the prompt.
        if last_user_text is None:
            raise ChatbotError("at least one user message is required", param="messages")

        rendered_lines.append("Assistant:")
        prompt = "\n".join(rendered_lines)
        return prompt, latest_image

    def _extract_content(self, content: Any, idx: int) -> Tuple[str, Optional[Image.Image]]:
        if content is None:
            return "", None
        if isinstance(content, str):
            return content.strip(), None
        if not isinstance(content, list):
            raise ChatbotError(
                f"messages[{idx}].content must be a string or array of content parts",
                param="messages",
            )
        text_chunks: List[str] = []
        image: Optional[Image.Image] = None
        for j, part in enumerate(content):
            if not isinstance(part, dict):
                raise ChatbotError(
                    f"messages[{idx}].content[{j}] must be an object",
                    param="messages",
                )
            ptype = part.get("type")
            if ptype == "text":
                t = part.get("text", "")
                if t:
                    text_chunks.append(str(t))
            elif ptype in {"image_url", "input_image"}:
                image = self._decode_image_url(part, idx, j)
            else:
                raise ChatbotError(
                    f"messages[{idx}].content[{j}].type '{ptype}' is not supported",
                    param="messages",
                )
        return " ".join(chunk.strip() for chunk in text_chunks if chunk.strip()), image

    @staticmethod
    def _decode_image_url(part: Dict[str, Any], idx: int, jdx: int) -> Image.Image:
        image_url = part.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
        else:
            url = image_url or part.get("url")
        if not isinstance(url, str) or not url:
            raise ChatbotError(
                f"messages[{idx}].content[{jdx}].image_url.url is required",
                param="messages",
            )

        raw: bytes
        if url.startswith("data:"):
            try:
                _, b64 = url.split(",", 1)
            except ValueError as exc:
                raise ChatbotError(
                    f"messages[{idx}].content[{jdx}].image_url.url is malformed data URI",
                    param="messages",
                ) from exc
            try:
                raw = base64.b64decode(b64)
            except Exception as exc:
                raise ChatbotError(
                    f"messages[{idx}].content[{jdx}].image_url.url base64 decode failed: {exc}",
                    param="messages",
                ) from exc
        else:
            scheme = urlparse(url).scheme.lower()
            if scheme not in {"http", "https"}:
                raise ChatbotError(
                    f"messages[{idx}].content[{jdx}].image_url.url must be a data URI or http(s) URL",
                    param="messages",
                )
            try:
                with urlopen(url, timeout=15) as resp:  # noqa: S310 - URL is user-supplied by API caller
                    raw = resp.read()
            except Exception as exc:
                raise ChatbotError(
                    f"failed to fetch image at {url}: {exc}",
                    param="messages",
                ) from exc

        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ChatbotError(
                f"could not decode image: {exc}",
                param="messages",
            ) from exc

    # ---------------------- generation ----------------------

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int,
                  temperature: float, do_sample: bool) -> str:
        self._ensure_loaded()
        torch = self._torch

        try:
            image_out = self._image_processor(images=[image], return_tensors="pt")
            pixel_values = image_out["pixel_values"] if isinstance(image_out, dict) else image_out.pixel_values
        except TypeError:
            pixel_values = self._image_processor(image).unsqueeze(0)
        pixel_values = pixel_values.to(self._device)

        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
                no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(
            output[0, input_ids.shape[1]:], skip_special_tokens=True,
        ).strip()


def build_chatbot_from_env() -> Optional[ChatbotService]:
    """Construct a ChatbotService from environment variables, or return None
    if not configured. ``GOPHER_EYE_LLM_CKPT`` is required; ``GOPHER_EYE_LLM_DIR``
    points at the ``LLM_work`` root so ``src.BLIP_Qwen`` is importable."""
    ckpt = os.environ.get("GOPHER_EYE_LLM_CKPT")
    if not ckpt:
        return None
    return ChatbotService(
        ckpt_dir=ckpt,
        llm_src_root=os.environ.get("GOPHER_EYE_LLM_DIR"),
        model_name=os.environ.get("GOPHER_EYE_LLM_MODEL_NAME", DEFAULT_MODEL_NAME),
        device=os.environ.get("GOPHER_EYE_LLM_DEVICE"),
    )
