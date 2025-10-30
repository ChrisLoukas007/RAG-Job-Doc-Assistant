import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing():
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:6006/v1/traces"
    )
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider = TracerProvider(
        resource=Resource.create({"service.name": "rag-job-doc-assistant"})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    print(f"Tracing configured to export to: {endpoint}")
