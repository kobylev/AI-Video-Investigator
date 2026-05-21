"""
WP5 - Reasoner & Router Integration
-----------------------------------
This module implements Agent 2 ("The Reasoner") of the AI Video Investigator's
two-stage cascade. The system follows a **Retrieve-then-Reason** architecture:

    Stage 1 (Retriever, WP4): FAISS + CLIP performs a cheap, dense vector
                              search over the entire video corpus and returns
                              a small set of candidate frames with similarity
                              scores.

    Stage 2 (Router, WP3):    A Confidence-Gated Router (BudgetAwareRouter)
                              partitions those candidates into three bands:
                                - ACCEPTED   (score >= tau_high)  -> trust CLIP
                                - AMBIGUOUS  (tau_low..tau_high)  -> escalate
                                - DISCARDED  (score <  tau_low)   -> drop

    Stage 3 (Reasoner, WP5):  THIS MODULE. Only the AMBIGUOUS subset is sent
                              to Gemini 1.5 Pro for high-order multimodal
                              reasoning. Every frame that bypasses Gemini is
                              a direct dollar-and-millisecond saving.

Token Economics Rationale
-------------------------
A naive baseline would send EVERY sampled frame (~3,600 frames for a 1-hour
1 FPS video) to Gemini, at ~258 image tokens + ~150 prompt tokens per call.
That is ~1.47M input tokens per hour of footage (= ~$5.15 at $3.50/M tokens),
plus latency on the order of minutes.

The Retrieve-then-Reason cascade reduces this to typically <5 frames per
query (the size of the AMBIGUOUS band after FAISS Top-K). Empirically this
yields a >99% token reduction with no measurable loss in Recall@10 on our
benchmark videos (see docs/WP4_FAISS_AND_EXPERIMENTS.md).
"""

from __future__ import annotations

import io
import os
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from PIL import Image
from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core import exceptions as gax_exceptions

# Load environment variables from .env at import time. This keeps API key
# handling consistent with the rest of the SDK (retriever, router) and avoids
# hard-coding secrets in source - a hard requirement for the academic
# security review (CISO compliance).
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasonerVerdict:
    """
    Immutable verdict object returned by the Reasoner.

    The frozen dataclass mirrors the JSON contract enforced on Gemini,
    giving the caller a strongly-typed handle instead of a raw dict.
    """
    event_detected: bool
    confidence_score: float
    reasoning: str
    raw_response: Optional[Any] = None  # underlying Gemini response (for metrics)

    @property
    def is_verified(self) -> bool:
        # Convenience flag used by main.py routing - keeps the call site clean.
        return self.event_detected and self.confidence_score >= 0.5


# ---------------------------------------------------------------------------
# Gemini Reasoner
# ---------------------------------------------------------------------------
class GeminiReasoner:
    """
    Agent 2 - Deep semantic verification using Gemini 1.5 Pro.

    OOP design notes
    ----------------
    * Single Responsibility: this class ONLY converts (frame, query) pairs
      into structured forensic verdicts. It does not know about FAISS, CLIP,
      thresholds, or the CLI - those concerns live in their own modules.
    * Dependency Inversion: the API key can be injected for tests, otherwise
      it is loaded from the environment.
    * Stateless verification: each `verify_event` call is independent, so the
      reasoner is safe to reuse across many ambiguous candidates in a loop.
    """

    # The SYSTEM_PROMPT is intentionally defined as a class constant so it
    # can be unit-tested, version-controlled, and audited as part of the
    # "Prompt Book" - a deliverable required by the academic defense.
    SYSTEM_PROMPT: str = (
        "You are an EXPERT FORENSIC VIDEO ANALYST embedded in an automated "
        "investigation pipeline. Your task is to inspect ONE or MORE still "
        "frames extracted from surveillance footage and decide whether the "
        "user's described event is visually occurring in any of them.\n\n"
        "ANALYTICAL GUIDELINES:\n"
        "  1. Reason ONLY from the visual evidence present in the frames. "
        "Do NOT infer events from context outside the image.\n"
        "  2. Treat ambiguity conservatively: if the evidence is partial or "
        "occluded, lower your confidence score accordingly.\n"
        "  3. Distinguish the queried event from visually similar but "
        "distinct activities (e.g. a person walking a dog vs. a dog chasing "
        "a person).\n"
        "  4. The reasoning string must cite the specific visual cues that "
        "drove your verdict (posture, motion blur, objects, spatial layout).\n\n"
        "OUTPUT CONTRACT (strict):\n"
        "  Return ONLY a JSON object - no prose, no markdown - matching:\n"
        "    {\n"
        '      "event_detected":   <boolean>,\n'
        '      "confidence_score": <float in [0.0, 1.0]>,\n'
        '      "reasoning":        <short string, <= 280 chars>\n'
        "    }"
    )

    # JSON Schema enforced by Gemini's `response_schema` parameter.
    # This is the structural-integrity contract: even if the model hallucinates
    # prose, the SDK will reject any response that does not match this shape,
    # which is critical for downstream parsing safety.
    _RESPONSE_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "event_detected":   {"type": "boolean"},
            "confidence_score": {"type": "number"},
            "reasoning":        {"type": "string"},
        },
        "required": ["event_detected", "confidence_score", "reasoning"],
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-1.5-pro-latest",
        max_image_dim: int = 512,
        max_retries: int = 3,
        retry_backoff_seconds: int = 30,
    ) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Add it to your .env file - the SDK "
                "refuses to start without an authenticated Reasoner."
            )

        # Configure the legacy `google.generativeai` SDK. We deliberately use
        # this SDK (not the newer google.genai Client) because its
        # `response_mime_type` + `response_schema` JSON-mode is the most
        # stable path to structurally-guaranteed output on Gemini 1.5 Pro.
        genai.configure(api_key=api_key)

        self.model_id = model_id
        self.max_image_dim = max_image_dim
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

        # Build the model once. The generation_config encodes our JSON
        # contract; system_instruction binds the SYSTEM_PROMPT at the
        # protocol layer rather than the message layer (cheaper, cached
        # on Google's side across calls).
        self._model = genai.GenerativeModel(
            model_name=self.model_id,
            system_instruction=self.SYSTEM_PROMPT,
            generation_config={
                "temperature": 0.1,          # near-deterministic forensic output
                "max_output_tokens": 512,
                "response_mime_type": "application/json",
                "response_schema": self._RESPONSE_SCHEMA,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def verify_event(
        self,
        frame: Image.Image,
        query: str,
    ) -> ReasonerVerdict:
        """
        Verify a single ambiguous frame.

        :param frame: PIL Image of the candidate frame extracted by the
                      Retriever. Will be downscaled before transmission to
                      cap per-call token cost (see _prepare_image).
        :param query: The original natural-language query from the user.
        :return:      A ReasonerVerdict with the structured Gemini output.

        Raises
        ------
        RuntimeError: when all retries are exhausted on a transient failure.
        google.api_core.exceptions.GoogleAPIError: on non-retryable errors
            (e.g. authentication, permission, malformed request).
        """
        return self.verify_events([frame], query)

    def verify_events(
        self,
        frames: List[Image.Image],
        query: str,
    ) -> ReasonerVerdict:
        """
        Batched verification - sends ALL ambiguous frames in a single call.

        Token Economics
        ---------------
        Multimodal inputs in Gemini share the same prompt + system tokens
        across all images in one request. Batching N ambiguous candidates
        into one call costs ~(N * image_tokens + 1 * prompt_tokens) instead
        of N * (image_tokens + prompt_tokens), saving (N-1) prompt copies.
        """
        if not frames:
            raise ValueError("verify_events called with no frames.")

        # Compose the user-turn message: image parts followed by the textual
        # query. We use a per-call "USER QUERY" tag to anchor the model's
        # attention to the question even if multiple frames are present.
        parts: List[Any] = [self._prepare_image(f) for f in frames]
        parts.append(
            f"USER QUERY: \"{query}\"\n"
            f"Number of frames provided: {len(frames)}.\n"
            "Apply the OUTPUT CONTRACT above."
        )

        response = self._call_with_retry(parts)

        # Parse the JSON. Even with response_schema enforcement, defensive
        # parsing keeps the call site safe if the SDK contract evolves.
        try:
            payload = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError) as e:
            raise RuntimeError(
                f"Gemini returned a non-JSON response despite schema "
                f"enforcement: {response!r}"
            ) from e

        return ReasonerVerdict(
            event_detected=bool(payload.get("event_detected", False)),
            confidence_score=float(payload.get("confidence_score", 0.0)),
            reasoning=str(payload.get("reasoning", "")).strip(),
            raw_response=response,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _prepare_image(self, frame: Image.Image) -> Image.Image:
        """
        Downscale to `max_image_dim` on the longest edge. Gemini bills each
        image at a fixed token count above 384px, so 512px is the academic
        sweet spot - enough fidelity for forensic cues, no wasted budget.
        """
        frame = frame.copy()
        frame.thumbnail((self.max_image_dim, self.max_image_dim))
        return frame

    def _call_with_retry(self, parts: List[Any]) -> Any:
        """
        Exponential-ish backoff over the well-known retryable Gemini errors:
          - ResourceExhausted (429): rate limit
          - ServiceUnavailable (503): transient backend
          - DeadlineExceeded (504): network/inference timeout

        Non-retryable errors (PermissionDenied, InvalidArgument,
        Unauthenticated) propagate immediately - they will never succeed on
        retry and must surface to the operator.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._model.generate_content(parts)

            except (
                gax_exceptions.ResourceExhausted,
                gax_exceptions.ServiceUnavailable,
                gax_exceptions.DeadlineExceeded,
            ) as e:
                last_exc = e
                wait = self.retry_backoff_seconds * attempt
                logger.warning(
                    "[Reasoner] Transient Gemini failure (%s). "
                    "Attempt %d/%d - sleeping %ds before retry.",
                    type(e).__name__, attempt, self.max_retries, wait,
                )
                time.sleep(wait)

            except gax_exceptions.PermissionDenied as e:
                # Most common cause: GEMINI_API_KEY lacks Generative Language
                # API access. Fail fast with an actionable message.
                raise RuntimeError(
                    "Gemini 1.5 Pro denied this API key (403). Enable the "
                    "'Generative Language API' in Google Cloud Console or "
                    "regenerate the key at https://aistudio.google.com/."
                ) from e

        raise RuntimeError(
            f"Gemini Reasoner exhausted {self.max_retries} retries."
        ) from last_exc
