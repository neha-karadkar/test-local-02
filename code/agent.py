import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import logging
import json
import re
from pathlib import Path

from modules.guardrails.content_safety_decorator import with_content_safety

from config import Config

# =========================
# GUARDRAILS CONFIGURATION
# =========================
GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

# =========================
# SYSTEM PROMPT & CONSTANTS
# =========================
SYSTEM_PROMPT = (
    "You are a professional assistant specializing in answering user questions by referencing a curated collection of resumes. "
    "Your task is to interpret the user's question, retrieve the most relevant information from the provided resume documents using Azure AI Search, "
    "and deliver a clear, concise, and factual answer. Only use information found in the retrieved context; do not speculate or fabricate details. "
    "If the answer cannot be found in the resumes, politely inform the user. Always maintain a professional and respectful tone, and ensure that no personal contact information is disclosed in your responses."
)
OUTPUT_FORMAT = (
    "Provide a direct, well-structured answer in complete sentences. If the question cannot be answered from the resumes, respond with a polite message indicating that the information is not available."
)
FALLBACK_RESPONSE = "I'm sorry, but I could not find the information you requested in the available resumes."

SELECTED_DOCUMENT_TITLES = [
    "resumes_collection4.pdf",
    "resumes_collection3.pdf",
    "resumes_collection1.pdf",
    "resumes_collection2.pdf"
]

ENRICHED_FIELDS = ["entities", "keyphrases", "relationships"]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# OBSERVABILITY LIFESPAN
# =========================
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(lifespan=_obs_lifespan,

    title="Resume Knowledge Answer Agent",
    description="Answers user questions by referencing a curated collection of resumes using Azure AI Search and GPT-4.1, with strict privacy and compliance controls.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

# =========================
# LOGGING CONFIGURATION
# =========================
_logger = logging.getLogger("agent")
_logger.setLevel(logging.INFO)

# =========================
# INPUT/OUTPUT MODELS
# =========================
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question about the resumes.")

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the query was processed successfully.")
    answer: Optional[str] = Field(None, description="The agent's answer to the user's question.")
    error: Optional[str] = Field(None, description="Error message if any.")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (always empty for this agent).")

# =========================
# LLM OUTPUT SANITIZER
# =========================
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# AZURE AI SEARCH CLIENT
# =========================
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

class AzureSearchClient:
    """Handles REST API calls to Azure AI Search endpoint, manages authentication, builds OData filters, parses search results."""

    def __init__(self):
        self.endpoint = Config.AZURE_SEARCH_ENDPOINT
        self.index_name = Config.AZURE_SEARCH_INDEX_NAME
        self.api_key = Config.AZURE_SEARCH_API_KEY
        self._client = None

    def get_client(self):
        if self._client is None:
            self._client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def search(self, query: str, embedding: List[float], filter_titles: List[str], top_k: int, select_fields: List[str]) -> List[dict]:
        """Perform vector + keyword search with OData filter on titles."""
        client = self.get_client()
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="vector")
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": select_fields,
        }
        if filter_titles:
            odata_parts = [f"title eq '{t}'" for t in filter_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)
        _t0 = _time.time()
        try:
            results = list(client.search(**search_kwargs))
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(results)[:200] if results is not None else None,
                    status="success",
                )
            except Exception:
                pass
            return results
        except Exception as e:
            try:
                trace_tool_call(
                    tool_name="search_client.search",
                    latency_ms=int((_time.time() - _t0) * 1000),
                    output=str(e),
                    status="error",
                    error=e,
                )
            except Exception:
                pass
            raise

# =========================
# CHUNK RETRIEVER
# =========================
class ChunkRetriever:
    """Queries Azure AI Search using vector + keyword search, applies OData filter for selected_document_titles, retrieves top_k relevant chunks."""

    def __init__(self, azure_search_client: AzureSearchClient):
        self.azure_search_client = azure_search_client
        self._enriched_available = None  # None = not yet checked, True/False after first search

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, filter_titles: List[str], top_k: int) -> List[str]:
        """Retrieve relevant chunks from Azure AI Search, with enriched fields fallback."""
        from azure.core.exceptions import HttpResponseError
        # Get embedding for the query
        embedding = await self._get_embedding(query)
        base_fields = ["chunk", "title"]
        select_fields = base_fields + ENRICHED_FIELDS if self._enriched_available is not False else base_fields
        client = self.azure_search_client

        # Try enriched fields, fallback to base fields if not available
        try:
            results = await client.search(query, embedding, filter_titles, top_k, select_fields)
            if self._enriched_available is None:
                self._enriched_available = True
                _logger.info("Enriched index fields are AVAILABLE — using: %s", ENRICHED_FIELDS)
            context_parts = []
            for r in results:
                part = r.get("chunk", "")
                if self._enriched_available:
                    for field in ENRICHED_FIELDS:
                        value = r.get(field)
                        if value:
                            part += f"\n{field}: {json.dumps(value) if isinstance(value, (list, dict)) else value}"
                context_parts.append(part)
            return context_parts
        except HttpResponseError as e:
            if "Could not find a property named" in str(e) and self._enriched_available is not False:
                self._enriched_available = False
                _logger.warning("Enriched index fields NOT available in this index — falling back to base fields: %s", base_fields)
                results = await client.search(query, embedding, filter_titles, top_k, base_fields)
                context_parts = [r.get("chunk", "") for r in results]
                return context_parts
            raise

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the query using Azure OpenAI."""
        client = get_llm_client()
        deployment = Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        _t0 = _time.time()
        resp = await client.embeddings.create(
            input=text,
            model=deployment
        )
        try:
            trace_model_call(
                provider="azure",
                model_name=deployment,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary="embedding"
            )
        except Exception:
            pass
        return resp.data[0].embedding

# =========================
# LLM SERVICE
# =========================
@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    api_key = Config.AZURE_OPENAI_API_KEY
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not configured")
    return openai.AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    )

class LLMService:
    """Calls Azure OpenAI GPT-4.1 with enhanced system prompt, user query, and retrieved chunks as context."""

    def __init__(self):
        self.model = Config.LLM_MODEL or "gpt-4.1"
        self.temperature = 0.7
        self.max_tokens = 2000

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_answer(self, prompt: str, context: List[str], tools: Optional[list] = None) -> str:
        """Call Azure OpenAI with system prompt, user query, and retrieved chunks."""
        client = get_llm_client()
        system_message = SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT
        context_text = "\n\n".join(context) if context else ""
        user_message = f"{prompt}\n\nContext:\n{context_text}" if context_text else prompt
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=self.model,
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            _logger.error(f"LLM API error: {e}")
            raise

# =========================
# PRIVACY REDACTOR
# =========================
class PrivacyRedactor:
    """Redacts or masks personal contact information (PII) such as email addresses and phone numbers from LLM responses."""

    EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b")

    def redact(self, response: str) -> str:
        """Redact email addresses and phone numbers from the response."""
        if not response:
            return response
        redacted = self.EMAIL_RE.sub("[EMAIL REDACTED]", response)
        redacted = self.PHONE_RE.sub("[PHONE REDACTED]", redacted)
        if redacted != response:
            _logger.info("PII redaction applied to LLM response.")
        return redacted

# =========================
# RESPONSE FORMATTER
# =========================
class ResponseFormatter:
    """Structures the final answer according to output_format instructions, applies fallback_response if needed."""

    def format(self, response: str) -> str:
        """Format the final answer, apply fallback if empty."""
        if not response or not response.strip():
            return FALLBACK_RESPONSE
        return response.strip()

# =========================
# TOOL REGISTRY (Framework Only)
# =========================
class BaseTool:
    """Base class for future tool integrations."""
    pass

class ToolRegistry:
    """Registers and manages OpenAI function-calling tools (none for this agent, but framework included for extensibility)."""

    def __init__(self):
        self._tools = []

    def register_tool(self, tool: BaseTool) -> None:
        self._tools.append(tool)

    def get_tools(self) -> list:
        return self._tools

# =========================
# ERROR HANDLER
# =========================
class ErrorHandler:
    """Handles timeouts, API errors, system errors, hard stops, and fallback behaviors."""

    def handle_error(self, error: Exception) -> str:
        _logger.error(f"Error handled: {error}")
        return FALLBACK_RESPONSE

# =========================
# AUDIT LOGGER
# =========================
class AuditLogger:
    """Records agent events, queries, responses, errors, and privacy actions for compliance and monitoring."""

    def log(self, event: dict) -> None:
        try:
            _logger.info(json.dumps(event, default=str))
        except Exception as e:
            _logger.warning(f"Audit log failed: {e}")

# =========================
# MAIN AGENT CLASS
# =========================
class ResumeKnowledgeAgent:
    """Coordinates the end-to-end flow: receives user query, invokes retrieval, assembles context, calls LLM, applies privacy redaction, formats response, manages errors and audit logs."""

    def __init__(self):
        self.azure_search_client = AzureSearchClient()
        self.chunk_retriever = ChunkRetriever(self.azure_search_client)
        self.llm_service = LLMService()
        self.privacy_redactor = PrivacyRedactor()
        self.response_formatter = ResponseFormatter()
        self.tool_registry = ToolRegistry()
        self.error_handler = ErrorHandler()
        self.audit_logger = AuditLogger()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def ask(self, query: str) -> str:
        """Receives user query, orchestrates retrieval, LLM call, privacy redaction, formatting, and returns answer."""
        async with trace_step(
            "retrieve_chunks",
            step_type="process",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"{len(r)} chunks" if isinstance(r, list) else "0 chunks"
        ) as step:
            try:
                chunks = await self.chunk_retriever.retrieve_chunks(
                    query=query,
                    filter_titles=SELECTED_DOCUMENT_TITLES,
                    top_k=5
                )
                step.capture(chunks)
                if not chunks:
                    self.audit_logger.log({
                        "event": "no_chunks_found",
                        "query": query,
                        "timestamp": _time.time()
                    })
                    return FALLBACK_RESPONSE
            except Exception as e:
                self.audit_logger.log({
                    "event": "retrieval_error",
                    "error": str(e),
                    "query": query,
                    "timestamp": _time.time()
                })
                return self.error_handler.handle_error(e)

        async with trace_step(
            "llm_generate_answer",
            step_type="llm_call",
            decision_summary="Generate answer from LLM using retrieved context",
            output_fn=lambda r: f"len={len(r) if r else 0}"
        ) as step:
            try:
                raw_llm_response = await self.llm_service.generate_answer(
                    prompt=query,
                    context=chunks,
                    tools=[]
                )
                step.capture(raw_llm_response)
            except Exception as e:
                self.audit_logger.log({
                    "event": "llm_error",
                    "error": str(e),
                    "query": query,
                    "timestamp": _time.time()
                })
                return self.error_handler.handle_error(e)

        async with trace_step(
            "sanitize_llm_output",
            step_type="process",
            decision_summary="Sanitize LLM output for formatting artefacts",
            output_fn=lambda r: f"len={len(r) if r else 0}"
        ) as step:
            sanitized = sanitize_llm_output(raw_llm_response, content_type="text")
            step.capture(sanitized)

        async with trace_step(
            "privacy_redaction",
            step_type="process",
            decision_summary="Redact PII from LLM response",
            output_fn=lambda r: f"len={len(r) if r else 0}"
        ) as step:
            redacted = self.privacy_redactor.redact(sanitized)
            step.capture(redacted)

        async with trace_step(
            "format_response",
            step_type="format",
            decision_summary="Format final answer and apply fallback if needed",
            output_fn=lambda r: f"len={len(r) if r else 0}"
        ) as step:
            formatted = self.response_formatter.format(redacted)
            step.capture(formatted)

        self.audit_logger.log({
            "event": "answer_generated",
            "query": query,
            "answer": formatted,
            "timestamp": _time.time()
        })
        return formatted

    def get_status(self) -> dict:
        """Return agent status."""
        return {
            "agent_name": getattr(Config, "AGENT_NAME", "Resume Knowledge Answer Agent"),
            "project_name": getattr(Config, "PROJECT_NAME", "dynamic-er"),
            "version": getattr(Config, "SERVICE_VERSION", "1.0.0"),
            "status": "ok"
        }

# =========================
# FASTAPI ENDPOINTS
# =========================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted.",
                "Check for missing commas, colons, or quotes.",
                "All required fields must be present and non-empty.",
                "Text fields must not exceed 50,000 characters."
            ]
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted.",
                "Check for missing commas, colons, or quotes.",
                "All required fields must be present and non-empty.",
                "Text fields must not exceed 50,000 characters."
            ]
        }
    )

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: QueryRequest):
    """Main query endpoint for resume knowledge agent."""
    agent = ResumeKnowledgeAgent()
    try:
        # Input validation
        if not req.query or not req.query.strip():
            return QueryResponse(
                success=False,
                answer=None,
                error="Query must not be empty.",
                tool_calls_made=[]
            )
        if len(req.query) > 50000:
            return QueryResponse(
                success=False,
                answer=None,
                error="Query exceeds maximum allowed length (50,000 characters).",
                tool_calls_made=[]
            )
        answer = await agent.ask(req.query.strip())
        return QueryResponse(
            success=True,
            answer=answer,
            error=None,
            tool_calls_made=[]
        )
    except Exception as e:
        _logger.error(f"Agent error: {e}")
        return QueryResponse(
            success=False,
            answer=None,
            error=str(e),
            tool_calls_made=[]
        )

# =========================
# MAIN ENTRYPOINT
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())