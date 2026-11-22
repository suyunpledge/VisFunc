"""SVSG HTTP 入口：依赖装配 + 请求封装 + 错误映射 + 访问留痕。

两种请求形态：
- ``POST /v1/analyze``：传入 L1 已产出的 IR 载荷（供 L1 与网关解耦部署）；
- ``POST /v1/analyze-image``：直接上传图片 + 查询文本，网关内部走
  ``image → L1 编译 → IR → 编排 → 答案`` 的完整闭环（应用化主形态）。

装配全部由 svsg.config.Settings 驱动（环境变量前缀 ``SVSG_`` + ``.env``）：
    SVSG_PORT=3001
    SVSG_LLM_PROVIDER=openai
    SVSG_LLM_BASE_URL / SVSG_LLM_API_KEY / SVSG_LLM_MODEL
    SVSG_DETECTOR=disabled|functions|demo|yolo   SVSG_YOLO_WEIGHTS=yolov8n.pt
    SVSG_SAMPLE_MODE=steady   SVSG_TRACE_FILE=traces.jsonl
    SVSG_API_KEY=...          # 设置后要求请求头 X-API-Key

语义约定：
- 业务级结果（rejected / human_review / delivered_with_review）以
  HTTP 200 + status 字段返回，由调用方按 status 分流；
- SVSGError 逃逸到 API 层视为编排器未消化的内部错误：
  E1xxx（参数类）→ 422，其余 → 500；
- 每个请求分配 trace_id（响应头 X-SVSG-Trace-Id），并在响应后
  按三档采样策略留痕（异常恒采样）。
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from hashlib import sha256
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from svsg import __version__
from svsg.auth import build_auth_router, require_bearer
from svsg.config import Settings, get_settings
from svsg.contracts import IR, SVSGError
from svsg.contracts.image_functions import MAX_RASTER_PIXELS, RASTER_QUERIES, RasterQuery
from svsg.contracts.recognition import FrameReference, Labels, RecognitionResult
from svsg.image_functions.compiler import FunctionImageCompiler
from svsg.l1_compiler import DemoDetector, ImageMeta, L1Compiler, YoloDetector, image_dimensions
from svsg.l3_orchestrator import (
    LLMProtocol,
    Orchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    StubLLM,
)
from svsg.l15_verifier import L15Verifier, VerifierConfig
from svsg.l15_verifier.openai_vision import OpenAIVisionVerifier
from svsg.observability import (
    SampleMode,
    SamplingPolicy,
    TraceStore,
    configure_logging,
    get_logger,
    get_trace_id,
    is_anomaly,
    new_trace_id,
)
from svsg.recognition import RecognitionService

logger = get_logger("api")

#: 上传图片字节上限(16MB):Starlette 不限制 multipart 大小,需自行设防
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = MAX_RASTER_PIXELS
MAX_QUERY_BYTES = 4096
EvidenceMode = Literal["submitted_ir", "image", "demo", "functions"]
ImageCompiler = L1Compiler | FunctionImageCompiler


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    ir: dict[str, Any] = Field(..., description="L1 产出的 IR 载荷（JSON）")


class AnalyzeResponse(BaseModel):
    status: str
    evidence_mode: EvidenceMode
    image_sha256: str | None = None
    final_answer: str | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)
    class_overrides: dict[str, str] = Field(default_factory=dict)
    degraded_instances: list[int] = Field(default_factory=list)
    disputed_instances: list[int] = Field(default_factory=list)
    verified_instances: list[int] = Field(default_factory=list)
    human_review_required: bool = False
    anchor_violations: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    session_summary: dict[str, Any] = Field(default_factory=dict)
    trace_id: str


# ---------------------------------------------------------------------- 装配


def build_llm(settings: Settings) -> LLMProtocol:
    """按配置装配 L3 后端；openai 模式缺 base_url 已在 Settings 校验期失败。"""
    if settings.llm_provider == "openai":
        from .l3_orchestrator.openai_adapter import OpenAICompatibleLLM

        assert settings.llm_base_url is not None
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    # Stub 后端无脚本答案 → complete 抛异常 → 编排器按 S5 中止。
    # 仅用于冒烟联调；生产必须切换 openai provider。
    return StubLLM()


def build_verifier(settings: Settings) -> L15Verifier:
    """Configured visual verification or explicit degradation."""
    config = VerifierConfig(timeout_s=settings.l15_timeout_s)
    if settings.l15_provider == "disabled":
        return L15Verifier(config=config)
    return L15Verifier(config=config, image_backend=OpenAIVisionVerifier(
        base_url=settings.l15_base_url,
        api_key=settings.l15_api_key,
        model=settings.l15_model,
        timeout_s=settings.l15_timeout_s,
        max_regions=settings.l15_max_regions,
    ))


def build_compiler(settings: Settings) -> ImageCompiler | None:
    """Only an explicitly selected detector accepts image uploads."""
    if settings.detector == "functions":
        return FunctionImageCompiler()
    if settings.detector == "yolo":
        return L1Compiler(YoloDetector(settings.yolo_weights))
    if settings.detector in ("demo", "stub"):
        return L1Compiler(DemoDetector())
    return None


def build_recognizer(settings: Settings) -> RecognitionService | None:
    if settings.vision_provider == "disabled":
        return None
    from svsg.recognition.adaptive import AdaptiveRecognitionService
    from svsg.recognition.provider import OpenAIRecognizer

    service_type = AdaptiveRecognitionService if settings.vision_strategy == "multiview" else (
        RecognitionService
    )
    budget = {"max_calls": settings.vision_max_calls,
              "discovery_count": settings.vision_discovery_views} if (
        settings.vision_strategy == "multiview"
    ) else {}
    return service_type(
        OpenAIRecognizer(
            base_url=settings.vision_base_url, api_key=settings.vision_api_key,
            model=settings.vision_model, timeout_s=settings.vision_timeout_s,
            thinking_disabled=settings.vision_thinking_disabled,
            json_mode=settings.vision_json_mode,
        ),
        reviewer=OpenAIRecognizer(
            base_url=settings.vision_base_url, api_key=settings.vision_api_key,
            model=settings.vision_review_model, timeout_s=settings.vision_timeout_s,
            thinking_disabled=settings.vision_thinking_disabled,
            json_mode=settings.vision_json_mode,
        ) if settings.vision_review_model else None,
        total_timeout_s=settings.vision_timeout_s, max_objects=settings.vision_max_objects,
        **budget,
    )


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """从配置装配编排器（L1.5/L3 超时预算由 Orchestrator 装配期断言）。"""
    settings = settings or get_settings()
    return Orchestrator(
        build_verifier(settings),
        build_llm(settings),
        OrchestratorConfig(llm_timeout_s=settings.llm_timeout_s),
    )


def _to_response(
    result: OrchestratorResult, evidence_mode: EvidenceMode, image_sha256: str | None = None
) -> AnalyzeResponse:
    """编排结果 → 统一响应体（analyze / analyze-image 共用）。"""
    return AnalyzeResponse(
        status="demo" if evidence_mode == "demo" and result.delivered else result.status,
        evidence_mode=evidence_mode,
        image_sha256=image_sha256,
        final_answer=result.answer.final_answer if result.answer else None,
        claims=[c.model_dump() for c in result.answer.claims] if result.answer else [],
        class_overrides={str(k): v for k, v in result.class_overrides.items()},
        degraded_instances=sorted(result.degraded_instances),
        disputed_instances=sorted(result.disputed_instances),
        verified_instances=sorted(result.verified_instances),
        human_review_required=result.human_review_required,
        anchor_violations=result.anchor_violations,
        error=result.error,
        session_summary=result.session_summary,
        trace_id=get_trace_id(),
    )


def _api_key_dependency(settings: Settings):
    """可选 API Key 鉴权依赖：settings.api_key 未设置时放行。"""

    async def require_api_key(request: Request) -> None:
        if settings.api_key is None:
            return
        provided = request.headers.get("X-API-Key")
        # 恒时比较:消除逐字符匹配的时序侧信道
        if not provided or not hmac.compare_digest(provided, settings.api_key):
            raise HTTPException(status_code=401, detail="无效或缺失的 API Key")

    return require_api_key


async def _read_image(file: UploadFile) -> tuple[bytes, ImageMeta, str]:
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="图片过大（超过 16MB）")
        chunks.append(chunk)
    data = b"".join(chunks)
    try:
        width, height = await asyncio.to_thread(image_dimensions, data)
    except ImportError as exc:  # pragma: no cover - Pillow is an API dependency
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="无法解析图片") from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="图像解码后的像素数超限")
    digest = sha256(data).hexdigest()
    return data, ImageMeta(image_id=f"sha256:{digest}", width=width, height=height), digest


def _parse_raster_queries(queries: str) -> list[RasterQuery]:
    if len(queries.encode("utf-8")) > MAX_QUERY_BYTES:
        raise HTTPException(status_code=413, detail="函数查询参数过长")
    try:
        return RASTER_QUERIES.validate_json(queries)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="无效的函数查询参数") from exc


async def _compile_functions(
    compiler: FunctionImageCompiler, image: bytes, meta: ImageMeta, queries: str
) -> IR:
    operations = _parse_raster_queries(queries)
    try:
        return await asyncio.to_thread(compiler.compile, image, meta, queries=operations)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"函数编译失败: {exc}") from exc


# ---------------------------------------------------------------------- 应用工厂


def create_app(
    orchestrator: Orchestrator | None = None,
    *,
    settings: Settings | None = None,
    compiler: ImageCompiler | None = None,
    recognizer: RecognitionService | None = None,
    screen_reader: Any | None = None,
) -> FastAPI:
    """应用工厂（测试注入编排器/编译器，生产走配置装配）。"""
    settings = settings or get_settings()
    configure_logging(level=getattr(logging, settings.log_level, logging.INFO))

    app = FastAPI(title="SVSG", version=__version__)
    app.state.settings = settings
    app.state.orchestrator = orchestrator or build_orchestrator(settings)
    app.state.compiler = compiler or build_compiler(settings)
    app.state.recognizer = recognizer or build_recognizer(settings)
    # 界面文字读取：OCR 模型首次请求时加载，之后常驻（每次加载约 2 秒）
    app.state.screen_reader = screen_reader
    screen_reader_lock = asyncio.Lock()
    app.state.trace_store = TraceStore(
        SamplingPolicy(mode=SampleMode(settings.sample_mode)),
        sink_path=settings.trace_file,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    require_api_key = _api_key_dependency(settings)
    if settings.auth_enabled:  # 登录机制 v0：挂载 /auth/* 并切换为 Bearer 鉴权
        app.include_router(build_auth_router(settings))
        require_api_key = require_bearer(settings)

    @app.middleware("http")
    async def trace_and_sample(request: Request, call_next):
        trace_id = new_trace_id()
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-SVSG-Trace-Id"] = trace_id
        summary = getattr(request.state, "session_summary", None)
        if summary is not None:
            violations = getattr(request.state, "anchor_violations", None) or []
            app.state.trace_store.record(
                {"path": request.url.path, "session": summary, "violations": violations,
                 "evidence_mode": request.state.evidence_mode},
                anomaly=is_anomaly(summary, anchor_violations=violations),
            )
        return response

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def analyze(req: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        result = await app.state.orchestrator.run(req.query, req.ir)
        request.state.evidence_mode = "submitted_ir"
        request.state.session_summary = result.session_summary
        request.state.anchor_violations = result.anchor_violations
        logger.info(
            "analyze.done", extra={"status": result.status, "query_len": len(req.query)}
        )
        return _to_response(result, "submitted_ir")

    @app.post(
        "/v1/compile-functions",
        response_model=IR,
        dependencies=[Depends(require_api_key)],
    )
    async def compile_functions(
        file: Annotated[UploadFile, File()],
        queries: Annotated[str, Form()] = "[]",
    ) -> IR:
        image, meta, _ = await _read_image(file)
        return await _compile_functions(FunctionImageCompiler(), image, meta, queries)

    @app.post(
        "/v1/recognize-image",
        response_model=RecognitionResult,
        dependencies=[Depends(require_api_key)],
    )
    async def recognize_image(
        request: Request,
        file: Annotated[UploadFile, File()],
        labels: Annotated[str, Form()] = "[]",
        frame: Annotated[str, Form()] = "null",
    ) -> RecognitionResult:
        service: RecognitionService | None = app.state.recognizer
        if service is None:
            raise HTTPException(status_code=503, detail="多模态识别后端未配置")
        if len(labels.encode()) + len(frame.encode()) > MAX_QUERY_BYTES:
            raise HTTPException(status_code=413, detail="识别参数过长")
        try:
            scope = TypeAdapter(Labels).validate_json(labels)
            reference: FrameReference | None = TypeAdapter(FrameReference | None).validate_json(
                frame
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="无效的识别类别或帧元数据") from exc
        data, _, _ = await _read_image(file)
        try:
            result = await service.recognize(data, labels=scope, frame=reference)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="无法编译单帧图片") from exc
        logger.info("recognize_image.done", extra={
            "status": result.status, "image_sha256": result.image_sha256,
            "objects": len(result.observations), "evidence_mode": "multimodal",
        })
        return result

    @app.post("/v1/screen-text", dependencies=[Depends(require_api_key)])
    async def screen_text(
        file: Annotated[UploadFile, File()],
        regions: Annotated[str, Form()] = "null",
    ) -> dict[str, Any]:
        """读出界面截图上的文字，带源像素框和点击点；regions 为 [[l,t,r,b], ...] 时只读这些区域。"""
        if len(regions.encode()) > MAX_QUERY_BYTES:
            raise HTTPException(status_code=413, detail="区域参数过长")
        try:
            boxes = TypeAdapter(list[tuple[int, int, int, int]] | None).validate_json(regions)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="无效的区域参数") from exc
        data, _, _ = await _read_image(file)
        async with screen_reader_lock:
            if app.state.screen_reader is None:
                try:
                    from svsg.recognition.screen_text import ScreenTextReader

                    app.state.screen_reader = await asyncio.to_thread(ScreenTextReader)
                except ImportError as exc:
                    raise HTTPException(
                        status_code=503, detail="界面文字读取需要安装 svsg[overlays]"
                    ) from exc
        try:
            result = await asyncio.to_thread(app.state.screen_reader.read, data, boxes)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"无法读取图片文字: {exc}") from exc
        logger.info("screen_text.done", extra={
            "lines": len(result["lines"]), "regions": len(result["regions_px"]),
            "elapsed_ms": result["elapsed_ms"],
        })
        return result

    @app.post(
        "/v1/analyze-image",
        response_model=AnalyzeResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def analyze_image(
        request: Request,
        query: Annotated[str, Form(min_length=1, max_length=2000)],
        file: Annotated[UploadFile, File()],
        queries: Annotated[str, Form()] = "[]",
    ) -> AnalyzeResponse:
        active_compiler: ImageCompiler | None = app.state.compiler
        if active_compiler is None:
            raise HTTPException(status_code=503, detail="图片检测器未配置")
        data, meta, digest = await _read_image(file)
        evidence_mode: EvidenceMode
        if isinstance(active_compiler, FunctionImageCompiler):
            evidence_mode = "functions"
            ir = await _compile_functions(active_compiler, data, meta, queries)
        else:
            if queries != "[]":
                raise HTTPException(status_code=422, detail="函数查询需要启用 functions 模式")
            evidence_mode = (
                "demo" if active_compiler.detector_model == DemoDetector.model_name else "image"
            )
            try:
                ir = await asyncio.to_thread(active_compiler.compile, data, meta)
            except SVSGError as err:
                return AnalyzeResponse(
                    status="rejected", evidence_mode=evidence_mode, image_sha256=digest,
                    error=err.to_dict(), trace_id=get_trace_id(),
                )

        image = data if evidence_mode == "image" else None
        result = await app.state.orchestrator.run(query, ir, image=image)
        response = _to_response(result, evidence_mode, digest)
        request.state.evidence_mode = evidence_mode
        request.state.session_summary = result.session_summary
        request.state.anchor_violations = result.anchor_violations
        logger.info(
            "analyze_image.done",
            extra={"status": response.status, "image_id": meta.image_id,
                   "evidence_mode": evidence_mode},
        )
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.exception_handler(SVSGError)
    async def svsg_error_handler(request: Request, exc: SVSGError) -> JSONResponse:
        # SVSGError 逃逸到 API 层 = 编排器未消化的内部错误
        status_code = 422 if exc.code.value.startswith("E1") else 500
        logger.error("svsg_error", extra={"code": exc.code.value})
        return JSONResponse(status_code=status_code, content={"error": exc.to_dict()})

    return app


app = create_app()
