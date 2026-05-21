"""
WP5 - Reasoner & Router Integration  (Anthropic backend)
--------------------------------------------------------
This module implements Agent 2 ("The Reasoner") of the AI Video Investigator's
two-stage cascade, using **Anthropic Claude Haiku 4.5** as the multimodal
forensic engine.

Architecture recap (Retrieve-then-Reason)
-----------------------------------------
    Stage 1 (Retriever, WP4): FAISS + CLIP - cheap dense vector search.
    Stage 2 (Router, WP3):    Confidence-Gated thresholding by tau_low/tau_high.
    Stage 3 (Reasoner, WP5):  THIS MODULE - Claude Haiku 4.5 verifies only
                              the AMBIGUOUS subset.

Token Economics Rationale
-------------------------
A naive baseline would send EVERY sampled frame (~3,600 frames per hour at
1 FPS) to a vision LLM. Claude Haiku 4.5 prices image+text at roughly
$1/M input and $5/M output tokens; an hour of footage at ~258 image tokens
per call would cost ~$1.00 per query plus tens of seconds of latency.

The Retrieve-then-Reason cascade reduces this to typically <5 frames per
query (the size of the AMBIGUOUS band after FAISS Top-K). Empirically this
yields a >99% token reduction with no measurable loss in Recall@10 on our
benchmark videos (see docs/WP4_FAISS_AND_EXPERIMENTS.md).

Why Claude Haiku 4.5?
---------------------
* Lowest-latency multimodal model in the Claude 4 family (sub-second on
  single-frame queries) - critical for an interactive CLI experience.
* Native tool-use / structured-output support: we force a `submit_verdict`
  tool call so the JSON schema is enforced by the API protocol, not by
  brittle prose-parsing of the model's free text.
* Pricing tier compatible with academic Free/Build credits.
"""

from __future__ import annotations

import io
import os
import time
import base64
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from PIL import Image
from dotenv import load_dotenv

import anthropic

# Load environment variables from .env at import time. Keeps API key handling
# consistent with the rest of the SDK and avoids hard-coding secrets in
# source - a hard requirement for the academic security review.
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasonerVerdict:
    """
    Immutable verdict object returned by the Reasoner.

    The frozen dataclass mirrors the JSON contract enforced on Claude via
    forced tool use, giving the caller a strongly-typed handle instead of a
    raw dict.
    """
    event_detected: bool
    confidence_score: float
    reasoning: str
    raw_response: Optional[Any] = None  # underlying anthropic Message (for metrics)

    @property
    def is_verified(self) -> bool:
        # Convenience flag used by main.py routing - keeps the call site clean.
        return self.event_detected and self.confidence_score >= 0.5


# ---------------------------------------------------------------------------
# Claude Reasoner
# ---------------------------------------------------------------------------
class ClaudeReasoner:
    """
    Agent 2 - Deep semantic verification using Anthropic Claude Haiku 4.5.

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

    # The SYSTEM_PROMPT is intentionally a class constant so it can be
    # unit-tested, version-controlled, and audited as part of the
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
        "  You MUST call the `submit_verdict` tool exactly once with the "
        "structured fields. Do not emit any other text."
    )

    # Tool schema enforced via Claude's `tool_choice={'type':'tool','name':...}`
    # forcing mechanism. Forcing the tool guarantees we receive a tool_use
    # block with a JSON-typed `input` matching this schema - no prose
    # parsing required.
    _VERDICT_TOOL: Dict[str, Any] = {
        "name": "submit_verdict",
        "description": (
            "Submit the forensic verdict for the supplied frame(s). "
            "Must be called exactly once."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_detected": {
                    "type": "boolean",
                    "description": "True iff the queried event is clearly occurring in any frame.",
                },
                "confidence_score": {
                    "type": "number",
                    "description": "Calibrated confidence in [0.0, 1.0].",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Short (<=280 chars) explanation citing the specific visual cues.",
                },
            },
            "required": ["event_detected", "confidence_score", "reasoning"],
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "claude-haiku-4-5-20251001",
        max_image_dim: int = 512,
        max_retries: int = 3,
        retry_backoff_seconds: int = 30,
    ) -> None:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is missing. Add it to your .env file - the "
                "SDK refuses to start without an authenticated Reasoner."
            )

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
        self.max_image_dim = max_image_dim
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

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
        :return:      A ReasonerVerdict with the structured Claude output.

        Raises
        ------
        RuntimeError: when all retries are exhausted on a transient failure,
                      or when the API denies the key (401/403).
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
        Multimodal Anthropic requests share the same prompt + system tokens
        across all images in one call. Batching N ambiguous candidates into
        one request costs ~(N * image_tokens + 1 * prompt_tokens) instead
        of N * (image_tokens + prompt_tokens), saving (N-1) prompt copies.
        """
        if not frames:
            raise ValueError("verify_events called with no frames.")

        # Compose the user-turn content blocks: image parts first, then the
        # textual anchor naming the user's query.
        content: List[Dict[str, Any]] = [self._image_block(f) for f in frames]
        content.append({
            "type": "text",
            "text": (
                f"USER QUERY: \"{query}\"\n"
                f"Number of frames provided: {len(frames)}.\n"
                "Apply the OUTPUT CONTRACT above by calling submit_verdict."
            ),
        })

        response = self._call_with_retry(content)
        payload = self._extract_tool_payload(response)

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
        Downscale to `max_image_dim` on the longest edge. Vision tokenization
        on Claude scales with pixel count, so 512px is the academic sweet
        spot - enough fidelity for forensic cues, no wasted budget.
        """
        frame = frame.copy()
        frame.thumbnail((self.max_image_dim, self.max_image_dim))
        return frame

    def _image_block(self, frame: Image.Image) -> Dict[str, Any]:
        """
        Encode a PIL frame as a base64 image content block in Anthropic's
        wire format.
        """
        buf = io.BytesIO()
        self._prepare_image(frame).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        }

    def _call_with_retry(self, content: List[Dict[str, Any]]) -> anthropic.types.Message:
        """
        Bounded retry over the well-known transient Anthropic failures:
          - RateLimitError (429)
          - APITimeoutError / APIConnectionError (network)
          - InternalServerError (5xx)

        Non-retryable errors (AuthenticationError, PermissionDeniedError,
        BadRequestError) propagate immediately - they will never succeed on
        retry and must surface to the operator.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._client.messages.create(
                    model=self.model_id,
                    max_tokens=512,
                    temperature=0.1,         # near-deterministic forensic output
                    system=self.SYSTEM_PROMPT,
                    tools=[self._VERDICT_TOOL],
                    tool_choice={"type": "tool", "name": self._VERDICT_TOOL["name"]},
                    messages=[{"role": "user", "content": content}],
                )

            except (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as e:
                last_exc = e
                wait = self.retry_backoff_seconds * attempt
                logger.warning(
                    "[Reasoner] Transient Claude failure (%s). "
                    "Attempt %d/%d - sleeping %ds before retry.",
                    type(e).__name__, attempt, self.max_retries, wait,
                )
                time.sleep(wait)

            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise RuntimeError(
                    "Anthropic denied this API key (401/403). Verify "
                    "ANTHROPIC_API_KEY at https://console.anthropic.com/."
                ) from e

        raise RuntimeError(
            f"Claude Reasoner exhausted {self.max_retries} retries."
        ) from last_exc

    @staticmethod
    def _extract_tool_payload(response: anthropic.types.Message) -> Dict[str, Any]:
        """
        Pull the structured `input` payload from the forced `submit_verdict`
        tool_use block. Forcing the tool means this block is guaranteed by
        the API contract, but we defensively raise if the SDK shape drifts.
        """
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_verdict":
                return dict(block.input)
        raise RuntimeError(
            f"Claude did not emit the expected submit_verdict tool_use block: {response!r}"
        )
