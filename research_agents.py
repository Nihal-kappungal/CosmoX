# research_agents.py
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
import fitz  # PyMuPDF
from pydantic import BaseModel, Field, HttpUrl
from rapidfuzz import fuzz, process

from agents import (
    Agent,
    Runner,
    ModelSettings,
    WebSearchTool,
    function_tool,
    FunctionTool,
)

# =========================
# Shared data contracts
# =========================

class PaperMeta(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[HttpUrl] = None

class PaperExtract(BaseModel):
    meta: PaperMeta
    abstract: str
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    results: str = ""
    limitations: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    text_snippets: list[str] = Field(default_factory=list)  # short, quoted snippets with page refs if known

class GapItem(BaseModel):
    severity: Literal["small", "medium", "extreme"]
    area: str                                  # e.g., "evaluation", "theory", "scalability"
    description: str
    evidence: list[str] = Field(default_factory=list)  # quoted snippets or citations
    risk_if_ignored: str
    suggested_fix_outline: str                 # 1–3 line outline, details come from Hypothesis agent

class GapReport(BaseModel):
    overall_novelty_score: float = Field(..., ge=0.0, le=1.0)  # 0..1 (higher = more novel)
    overlap_refs: list[str] = Field(default_factory=list)      # URLs/DOIs considered overlapping
    gaps: list[GapItem] = Field(default_factory=list)

class HypothesisItem(BaseModel):
    name: str
    statement: str
    rationale: str
    test_design: str         # brief experimental design
    success_metrics: list[str]
    data_needed: list[str]
    estimated_scope: Literal["S", "M", "L"]

class HypothesisPlan(BaseModel):
    for_paper: str
    prioritized: list[HypothesisItem]
    milestone_plan: list[str]  # week-by-week or phase bullets


# =========================
# Low-level utility tools
# =========================

@function_tool
def extract_text_from_pdf(path: str) -> str:
    """
    Extract raw text from a local PDF path using PyMuPDF.

    Args:
        path: Absolute or relative path to a PDF file.
    """
    p = Path(path)
    if not p.exists():
        return f"[ERROR] File not found: {path}"
    doc = fitz.open(p.as_posix())
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            chunks.append(f"[PAGE {i+1}]\n{text}")
    return "\n\n".join(chunks) if chunks else "[WARN] No text extracted."

@function_tool
def download_pdf(url: str, save_dir: str = "downloads") -> str:
    """
    Download a PDF from URL and save locally. Returns saved path (or error).

    Args:
        url: Direct link to a PDF.
        save_dir: Directory to save into (created if missing).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", url.split("/")[-1] or "paper.pdf")
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    path = Path(save_dir) / filename
    try:
        with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as r:
            r.raise_for_status()
            # quick content-type check
            ctype = r.headers.get("content-type", "")
            if "pdf" not in ctype and not url.lower().endswith(".pdf"):
                return f"[ERROR] URL does not look like a PDF: {url} (content-type={ctype})"
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPError as e:
        return f"[ERROR] Download failed: {e}"
    return path.as_posix()

@function_tool
def rough_overlap_score(a: list[str], b: list[str]) -> float:
    """
    Compute a rough 0..1 overlap score between two sets of keywords using fuzzy matching.
    Intended only as a heuristic to help the Gap Detector.

    Args:
        a: list of key terms from focal paper
        b: list of key terms from related literature
    """
    if not a or not b:
        return 0.0
    scores = []
    for term in a:
        best = process.extractOne(term, b, scorer=fuzz.token_set_ratio)
        if best:
            scores.append(best[1] / 100.0)
    return float(sum(scores) / max(1, len(a)))


# =========================
# Agent: Research Fetcher (scouting + acquisition)
# =========================

fetcher_agent = Agent(
    name="Research Fetcher",
    instructions=(
        "You are a meticulous research scout. Given a topic or paper, "
        "you (1) find high-quality, recent sources; (2) identify the top 3–8 "
        "relevant papers with stable links; and (3) when asked, download PDFs. "
        "Prefer official publisher, arXiv, ACL Anthology, IEEE, ACM, or NIPS links. "
        "Return results as concise bullets with title, year, venue, URL, and a one-line relevance note."
    ),
    tools=[
        WebSearchTool(),      # hosted tool (Responses API)
        download_pdf,
    ],
    # You can tune the agent model if you like; otherwise it uses the SDK default.
    model_settings=ModelSettings(temperature=0.2),
)

# =========================
# Agent: Knowledge Extractor (paper -> structured)
# =========================

@function_tool
def normalize_paper_extract(meta_json: str, full_text: str) -> str:
    """
    Normalize a paper into a structured JSON 'PaperExtract'.

    Args:
        meta_json: a JSON object with rough metadata fields if you have them; may be empty '{}'.
        full_text: full raw text from PDF or copy-paste; include abstract if possible.
    """
    # This is a shim. The LLM will handle the heavy lifting and return JSON.
    # We expose this as a tool to let the model delegate formatting to a deterministic step if needed.
    try:
        meta = json.loads(meta_json) if meta_json else {}
    except Exception:
        meta = {}
    pe = PaperExtract(
        meta=PaperMeta(**meta) if meta else PaperMeta(title="Untitled"),
        abstract=full_text[:1000] if full_text else "",
    )
    return pe.model_dump_json(indent=2)

extractor_agent = Agent(
    name="Knowledge Extractor",
    instructions=(
        "You transform research papers into structured JSON (PaperExtract). "
        "Extract: title/authors/year/venue/doi/url, abstract, main methods, datasets, key results, "
        "limitations, contributions, and 5–15 domain key_terms. "
        "Quote 3–8 short text_snippets with page markers when possible. "
        "When given a PDF path, call extract_text_from_pdf first. "
        "Always return a valid JSON object conforming to PaperExtract."
    ),
    tools=[extract_text_from_pdf, normalize_paper_extract],
    model_settings=ModelSettings(temperature=0.1),
)

# =========================
# Agent: Domain Expert (adaptive by topic)
# =========================

domain_expert_agent = Agent(
    name="Domain Expert",
    instructions=(
        "You act as a senior domain researcher across CS subfields (ML, CV, NLP, Systems, SE, HCI, Security, DB). "
        "You ground your analysis with (a) textbook-level priors, (b) canonical methods, and (c) the latest results. "
        "For any niche term, briefly define it first time. Cite URLs you found via web search. "
        "Your outputs are pragmatic and oriented to experiments, datasets, baselines, ablations, and SOTA comparisons."
    ),
    tools=[WebSearchTool()],
    model_settings=ModelSettings(temperature=0.2),
)

# =========================
# Agent: Gap Detector (small/medium/extreme)
# =========================

gap_detector_agent = Agent(
    name="Gap Detector",
    instructions=(
        "You are a rigorous gap analyst. Given one focal PaperExtract and 3–12 related PaperExtracts or summaries, "
        "identify gaps at three severities:\n"
        "- small: missing ablations, unclear reporting, dataset leakage, weak error analysis\n"
        "- medium: limited generalization, missing baselines, weak theoretical link, reproducibility issues\n"
        "- extreme: flawed problem framing, confounded evaluation, unrealistic assumptions, lack of causal evidence, "
        "or novelty already solved elsewhere.\n\n"
        "Compute overall_novelty_score in [0,1] where 0=no novelty and 1=strong novelty. "
        "Use rough_overlap_score on key_terms as a heuristic when related keywords are provided. "
        "List overlap_refs (URLs/DOIs) that most overlap. "
        "Return JSON conforming to GapReport."
    ),
    tools=[rough_overlap_score, WebSearchTool()],
    model_settings=ModelSettings(temperature=0.1),
)

# =========================
# Agent: Hypothesis Generator (gap -> plan)
# =========================

hypothesis_agent = Agent(
    name="Hypothesis Generator",
    instructions=(
        "You are a proposal architect. Given a GapReport and Domain Expert notes, "
        "propose 2–6 concrete hypotheses to close the gaps. "
        "Each hypothesis: a precise statement, short rationale, test design (data, baselines, controls, ablations), "
        "success metrics (statistical tests if relevant), data_needed, and estimated_scope (S/M/L). "
        "Prioritize by expected impact / feasibility and output a HypothesisPlan JSON."
    ),
    model_settings=ModelSettings(temperature=0.2),
)

# =========================
# Orchestrator: Advanced Researcher
# =========================
# We present sub-agents to the orchestrator *as tools*, which is the recommended pattern for
# multi-agent coordination when you want one chat box controlling everything.

orchestrator = Agent(
    name="Advanced Researcher",
    instructions=(
        "You are the coordinator. Read the user's request and decide which specialized tool(s) to call. "
        "Typical flows:\n"
        "• User gives topic ➜ call Research Fetcher ➜ pick 3–8 top papers ➜ (optional) download ➜ pass to Extractor.\n"
        "• User uploads PDFs ➜ call Extractor to produce PaperExtract JSONs.\n"
        "• For critique ➜ call Domain Expert for background + baselines.\n"
        "• For gap analysis ➜ call Gap Detector with focal + related extracts.\n"
        "• For solutions ➜ call Hypothesis Generator with GapReport + domain notes.\n\n"
        "Always return a short executive summary for the user first, then attach the JSON payload(s)."
    ),
    tools=[
        fetcher_agent.as_tool(
            tool_name="scout_papers",
            tool_description="Search the web for recent & relevant papers; can download PDFs on request.",
        ),
        extractor_agent.as_tool(
            tool_name="extract_paper",
            tool_description="Convert a paper (text or local PDF path) into a structured JSON (PaperExtract).",
        ),
        domain_expert_agent.as_tool(
            tool_name="domain_expertise",
            tool_description="Provide domain-grounded critique, definitions, baselines, and SOTA pointers.",
        ),
        gap_detector_agent.as_tool(
            tool_name="detect_gaps",
            tool_description="Analyze focal and related papers to produce a GapReport JSON with severity levels.",
        ),
        hypothesis_agent.as_tool(
            tool_name="design_hypotheses",
            tool_description="Turn a GapReport into prioritized, testable hypotheses and a milestone plan.",
        ),
    ],
    model_settings=ModelSettings(temperature=0.2),
)

# =========================
# High-level convenience functions for your UI layer
# =========================

@dataclass
class RunArtifacts:
    summary: str
    payloads: dict[str, Any]  # any JSONs emitted by sub-agents

async def run_topic_to_gaps_and_plan(topic: str) -> RunArtifacts:
    """
    End-to-end flow from topic -> papers -> extracts -> gaps -> hypothesis plan.
    You can call sub-steps individually in your chat UI too.
    """
    payloads: dict[str, Any] = {}

    # 1) Scout
    res1 = await Runner.run(orchestrator, input=f"Find recent top papers and links for: {topic}. Then stop.")
    payloads["scout"] = res1.final_output

    # 2) (Optional) Download + Extract (demo: we only extract from URLs if Direct PDFs)
    # In a real app: iterate results, call scout_papers + download_pdf, then extract_paper per file
    # Here we just show how you would call the extractor explicitly if you had a path:
    # res_pdf = await Runner.run(orchestrator, input={'tool': 'extract_paper', 'path': 'downloads/paper.pdf'})

    # 3) Ask the orchestrator to produce gaps & plan directly (it will call sub-tools with context)
    res2 = await Runner.run(orchestrator, input=(
        "Given the scouted results, analyze likely gaps (small, medium, extreme) "
        "and propose hypotheses with a milestone plan. Return JSON objects."
    ))
    payloads["analysis"] = res2.final_output

    return RunArtifacts(summary="Completed topic → gaps → plan pipeline.", payloads=payloads)


# If you want a simple CLI demo:
if __name__ == "__main__":
    async def _demo():
        topic = "Domain adaptation for multimodal LLMs under distribution shift"
        arts = await run_topic_to_gaps_and_plan(topic)
        print(arts.summary)
        print("\n=== OUTPUTS ===\n")
        for k, v in arts.payloads.items():
            print(f"[{k}]\n{v}\n")
    asyncio.run(_demo())
