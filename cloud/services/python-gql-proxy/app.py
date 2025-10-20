import os
import logging
from typing import Dict

import requests
from flask import Flask, request, Response


def _copy_headers(src: Dict[str, str]) -> Dict[str, str]:
    # Forward most headers except hop-by-hop and content-length (requests sets it)
    excluded = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
    return {k: v for k, v in src.items() if k.lower() not in excluded}


def create_app() -> Flask:
    app = Flask(__name__)

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("python-gql-proxy")

    upstream_url = os.getenv("UPSTREAM_GRAPHQL_URL")
    if not upstream_url:
        logger.warning("UPSTREAM_GRAPHQL_URL is not set; service will return 500 for proxy requests")

    session = requests.Session()
    timeout_s = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "10"))

    @app.get("/health")
    def health() -> Response:
        return Response("ok", status=200, content_type="text/plain")

    @app.post("/graphql")
    def proxy_graphql() -> Response:
        nonlocal upstream_url
        if not upstream_url:
            return Response(
                "UPSTREAM_GRAPHQL_URL not configured",
                status=500,
                content_type="text/plain",
            )

        try:
            incoming_json = request.get_json(silent=True)
            if incoming_json is None:
                return Response("Invalid JSON body", status=400, content_type="text/plain")

            fwd_headers = _copy_headers(dict(request.headers))
            fwd_headers.setdefault("content-type", "application/json")

            logger.info(
                "proxy request",
                extra={
                    "method": request.method,
                    "path": request.path,
                    "upstream": upstream_url,
                },
            )

            upstream_resp = session.post(
                upstream_url,
                json=incoming_json,
                headers=fwd_headers,
                timeout=timeout_s,
            )

            # Prepare response with status, headers, and body
            resp_headers = {}
            # Preserve common caching/content headers
            for k, v in upstream_resp.headers.items():
                kl = k.lower()
                if kl in ("content-type", "cache-control", "etag", "expires"):  # safe set
                    resp_headers[k] = v

            return Response(
                upstream_resp.content,
                status=upstream_resp.status_code,
                headers=resp_headers,
            )
        except requests.Timeout:
            logger.warning("upstream timeout", extra={"upstream": upstream_url})
            return Response("Upstream timeout", status=504, content_type="text/plain")
        except requests.RequestException as e:
            logger.error("upstream error", extra={"error": str(e)})
            return Response("Upstream error", status=502, content_type="text/plain")
        except Exception as e:  # noqa: BLE001
            logger.exception("proxy handler error: %s", e)
            return Response("Internal error", status=500, content_type="text/plain")

    # Optional: Cloud Functions (2nd gen) entrypoint compatibility
    # The below function can be referenced as the entrypoint when deploying
    # a Functions target instead of Cloud Run.
    def cloud_function(request_):
        with app.request_context(request_.environ):
            return proxy_graphql()

    app.cloud_function = cloud_function  # type: ignore[attr-defined]
    return app


app = create_app()

if __name__ == "__main__":  # For local debug
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))


