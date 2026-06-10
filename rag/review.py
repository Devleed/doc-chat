import json
import os
import re
from io import BytesIO
from typing import TypedDict

import docx
from openai import OpenAI
from pypdf import PdfReader

REVIEW_SYSTEM_PROMPT = """
You are a strict project requirements validator. Given a project document, check whether it contains all of the following required fields:

1. Project Description — what problem is being solved and the overall scope
2. Milestones — concrete checkpoints with titles and descriptions
3. Requirements — functional and/or non-functional requirements
4. Deliverables — tangible outputs handed over at each stage
5. Tech Stack — the technologies, frameworks, or platforms to be used
6. Timeline — overall project duration or schedule
7. Cost / Budget — total cost or budget breakdown

Return ONLY valid JSON (no markdown fences) matching this exact schema:

{
  "approved": <true if ALL fields are present, false otherwise>,
  "errors": [
    {
      "field": "<field name>",
      "message": "<specific error describing exactly what is missing in THIS document>",
      "reason": "<why this field is required for a valid project requirements document>"
    }
  ],
  "milestones": [
    {
      "title": "<string>",
      "description": "<string>",
      "startDate": "<ISO date string or null>",
      "endDate": "<ISO date string or null>",
      "amount": "<string e.g. '2.5 SOL' or null>"
    }
  ]
}

Rules:
- "errors" must contain one entry for every field that is absent or insufficiently defined. If all fields are present, "errors" must be an empty array.
- "milestones" must be extracted from the document if present; otherwise use an empty array.
- Each "message" must be specific to what was actually found (or not found) in this document — do not use generic boilerplate.
- Each "reason" must explain concisely why that field is critical for a project requirements contract.
- "approved" must be true only when "errors" is empty.
- Return ONLY the JSON object. No explanation outside it.
""".strip()


class FieldError(TypedDict):
    field: str
    message: str
    reason: str


class Milestone(TypedDict):
    title: str
    description: str
    startDate: str | None
    endDate: str | None
    amount: str | None


class ReviewResult(TypedDict):
    approved: bool
    errors: list[FieldError]
    milestones: list[Milestone]
    error: str | None  # set only when processing itself fails


def _extract_text(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)
    if ext in {".txt", ".md"}:
        return file_bytes.decode("utf-8", errors="replace")
    if ext == ".docx":
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ""


def _call_openai(raw_text: str, client: OpenAI) -> dict | None:
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```[a-z]*\n?", "", content.strip())
    content = re.sub(r"\n?```$", "", content.strip())
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def review_doc(file_bytes: bytes, filename: str) -> ReviewResult:
    raw_text = _extract_text(file_bytes, filename)
    if not raw_text:
        return ReviewResult(
            approved=False, errors=[], milestones=[],
            error="Could not extract text from the uploaded file.",
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY"))

    result = _call_openai(
        f"Document filename: {filename}\n\nDocument content:\n{raw_text}", client
    )
    if result is None:
        return ReviewResult(
            approved=False, errors=[], milestones=[],
            error="Failed to parse the document. Please ensure the file is a valid project requirements document.",
        )

    errors: list[FieldError] = [
        FieldError(
            field=e.get("field", "Unknown"),
            message=e.get("message", ""),
            reason=e.get("reason", ""),
        )
        for e in (result.get("errors") or [])
    ]

    milestones: list[Milestone] = [
        Milestone(
            title=m.get("title") or "",
            description=m.get("description") or "",
            startDate=m.get("startDate") or None,
            endDate=m.get("endDate") or None,
            amount=m.get("amount") or None,
        )
        for m in (result.get("milestones") or [])
    ]

    approved = len(errors) == 0

    return ReviewResult(
        approved=approved,
        errors=errors,
        milestones=milestones,
        error=None,
    )
