"""FastAPI implementation of the Gopher Eye HTTP API.

The HTTP contract (paths, methods, status codes, response shapes) matches the
prior Flask implementation. The OpenAI-compatible ``/v1/models`` and
``/v1/chat/completions`` endpoints preserve their OpenAI error envelope and
reject ``stream=true`` with a 400 (non-streaming only).
"""

from __future__ import annotations

import io
import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from chatbot import ChatbotError
from samples_service import SamplesError


def _samples_or_503(samples_service):
    if samples_service is not None:
        return None
    return JSONResponse(
        {
            "error": {
                "message": "samples are not configured on this server",
                "code": "samples_disabled",
            }
        },
        status_code=503,
    )


def _samples_error(err: SamplesError) -> JSONResponse:
    return JSONResponse(err.to_payload(), status_code=err.status_code)


def _parse_csv_query(value: Optional[str]):
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _is_multipart(request: Request) -> bool:
    return (request.headers.get("content-type") or "").lower().startswith(
        "multipart/form-data"
    )


def _is_json(request: Request) -> bool:
    return (request.headers.get("content-type") or "").lower().startswith(
        "application/json"
    )


def create_api(name: str, application_layer=None, chatbot=None, samples_service=None) -> FastAPI:
    server = FastAPI(title=name)

    # ---------------- vpn (ZeroTier broker proxy) ----------------
    # Imported lazily so the rest of the API stays usable in environments
    # that haven't configured the broker (e.g. local dev without VPN).
    if os.environ.get("BROKER_URL"):
        from vpn_routes import build_router as build_vpn_router
        server.include_router(build_vpn_router())

    # ---------------- segmentation / plants / trials ----------------

    @server.put("/dl/segmentation")
    async def segment_plant(request: Request):
        if not _is_multipart(request):
            print(f"{request.headers.get('content-type')} is not 'multipart/form-data'")
            raise HTTPException(status_code=400)
        form = await request.form()
        if "image" not in form:
            raise HTTPException(status_code=400)
        data = {}
        if "data" in form:
            data = json.loads(form["data"])
        image_bytes = await form["image"].read()
        return {
            "plant_id": application_layer.segment_plant(image_bytes, data=data)
        }

    @server.put("/dl/segmentation_spike")
    async def segment_spike(request: Request):
        if not _is_multipart(request):
            print(f"{request.headers.get('content-type')} is not 'multipart/form-data'")
            raise HTTPException(status_code=400)
        form = await request.form()
        if "image" not in form:
            raise HTTPException(status_code=400)
        data = {}
        if "data" in form:
            data = json.loads(form["data"])
        image_bytes = await form["image"].read()
        return {
            "plant_id": application_layer.segment_plant(
                image_bytes, task="spike", data=data
            )
        }

    @server.get("/trials")
    async def get_trial():
        return application_layer.get_trials()

    @server.post("/trial")
    async def create_trial(request: Request):
        if not _is_json(request):
            print(f"{request.headers.get('content-type')} is not 'application/json'")
            raise HTTPException(status_code=400)
        trial_data = await request.json()
        trial_id = application_layer.create_trial(trial_data)
        return {"trial_id": trial_id}

    @server.put("/perf/segmentation")
    async def perf_test_segmentation(request: Request):
        if not _is_multipart(request):
            print(f"{request.headers.get('content-type')} is not 'multipart/form-data'")
            raise HTTPException(status_code=400)
        form = await request.form()
        if "image" not in form:
            raise HTTPException(status_code=400)
        image_bytes = await form["image"].read()
        plant_id = application_layer.segment_plant(image_bytes)
        image_data, mimetype = application_layer.get_image(plant_id, "segmentation")
        content = image_data.read() if hasattr(image_data, "read") else image_data
        return Response(content=content, media_type=mimetype)

    @server.get("/plant/status")
    async def get_plant_status(plant_id: Optional[str] = None):
        return {"status": application_layer.plant_status(plant_id)}

    @server.get("/plant/data")
    async def get_plant_data(plant_id: Optional[str] = None):
        return application_layer.plant_data(plant_id)

    @server.get("/plant/image")
    async def get_plant_item(plant_id: Optional[str] = None, image_name: Optional[str] = None):
        image_data, mimetype = application_layer.get_image(plant_id, image_name)
        if not image_data:
            raise HTTPException(status_code=400)
        content = image_data.read() if hasattr(image_data, "read") else image_data
        return Response(content=content, media_type=mimetype)

    @server.get("/plant/ids")
    async def get_plant_ids():
        return {"plant_ids": application_layer.get_plant_ids()}

    # ---------------- mock auth stubs ----------------

    @server.post("/register")
    async def register(status: Optional[str] = None):
        if status == "200":
            return JSONResponse(
                {
                    "status": 200,
                    "message": "OTP sent successfully to your email please check your email!",
                },
                status_code=200,
            )
        if status == "201":
            return JSONResponse(
                {
                    "status": 201,
                    "message": "OTP sent successfully to your email please check your email!",
                },
                status_code=201,
            )
        if status == "401":
            return JSONResponse(
                {"status": 401, "message": "Something went wrong"}, status_code=401
            )
        return JSONResponse(
            {
                "status": 200,
                "message": "OTP sent successfully to your email please check your email!",
            },
            status_code=200,
        )

    @server.post("/otpVerification")
    async def verify_otp(status: Optional[str] = None):
        if status == "200":
            return JSONResponse(
                {
                    "status": 200,
                    "message": "OTP verified successfully",
                    "token": "test_token",
                },
                status_code=200,
            )
        if status == "201":
            return JSONResponse(
                {
                    "status": 201,
                    "message": "OTP verified successfully",
                    "token": "test_token",
                },
                status_code=201,
            )
        if status == "401":
            return JSONResponse(
                {"status": 401, "message": "OTP is not valid"}, status_code=401
            )
        return JSONResponse(
            {
                "status": 200,
                "message": "OTP verified successfully",
                "token": "test_token",
            },
            status_code=200,
        )

    @server.post("/signin")
    async def signin(status: Optional[str] = None):
        if status == "200":
            return JSONResponse(
                {
                    "status": 200,
                    "message": "login successfully",
                    "token": "test_token",
                },
                status_code=200,
            )
        if status == "201":
            return JSONResponse(
                {
                    "status": 201,
                    "message": "login successfully",
                    "token": "test_token",
                },
                status_code=201,
            )
        if status == "401":
            return JSONResponse(
                {
                    "status": 401,
                    "message": "email or password is wrong/something went wrong",
                },
                status_code=401,
            )
        return JSONResponse(
            {
                "status": 200,
                "message": "login successfully",
                "token": "test_token",
            },
            status_code=200,
        )

    @server.get("/status")
    async def get_status():
        return {"status": "ok"}

    # ---------------- Mobile-aligned samples / instances / chat ----------------
    # See ``samples_service.py`` for the canonical schema; field names mirror
    # ``mobile-app/lib/services/sample_repository.dart``. BLOBs are not inlined
    # by default — list endpoints return ``has_<field>`` booleans and clients
    # download blobs via the dedicated ``/blob`` routes (or pass
    # ``include_blobs=*`` to embed base64 strings inline).

    @server.get("/samples")
    async def list_samples(request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        try:
            since = request.query_params.get("since")
            limit = request.query_params.get("limit")
            samples = samples_service.list_samples(
                user=request.query_params.get("user") or None,
                detection_mode=request.query_params.get("detection_mode") or None,
                since=int(since) if since else None,
                limit=int(limit) if limit else None,
                include_blobs=_parse_csv_query(request.query_params.get("include_blobs")),
            )
        except SamplesError as err:
            return _samples_error(err)
        except ValueError as err:
            return JSONResponse(
                {"error": {"message": str(err), "code": "invalid_query"}},
                status_code=400,
            )
        return {"samples": samples}

    @server.post("/samples")
    async def create_sample(request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not _is_multipart(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be multipart/form-data with image + data fields",
                        "code": "invalid_content_type",
                    }
                },
                status_code=400,
            )
        form = await request.form()
        if "image" not in form:
            return JSONResponse(
                {"error": {"message": "image file is required", "code": "missing_image"}},
                status_code=400,
            )
        upload = form["image"]
        image_bytes = await upload.read()
        metadata = {}
        if "data" in form:
            try:
                metadata = json.loads(form["data"])
            except json.JSONDecodeError as err:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"data field is not valid JSON: {err}",
                            "code": "invalid_json",
                        }
                    },
                    status_code=400,
                )
        try:
            sample = samples_service.create_sample(
                image_bytes, metadata, filename=getattr(upload, "filename", None)
            )
        except SamplesError as err:
            return _samples_error(err)
        return JSONResponse(sample, status_code=201)

    @server.get("/samples/{sample_id}")
    async def get_sample(sample_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        try:
            sample = samples_service.get_sample(
                sample_id,
                include_blobs=_parse_csv_query(request.query_params.get("include_blobs")),
            )
        except SamplesError as err:
            return _samples_error(err)
        if sample is None:
            return JSONResponse(
                {"error": {"message": "sample not found", "code": "not_found"}},
                status_code=404,
            )
        return sample

    @server.patch("/samples/{sample_id}")
    async def patch_sample(sample_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not _is_json(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be application/json",
                        "code": "invalid_content_type",
                    }
                },
                status_code=400,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        try:
            sample = samples_service.update_sample(sample_id, body or {})
        except SamplesError as err:
            return _samples_error(err)
        if sample is None:
            return JSONResponse(
                {"error": {"message": "sample not found", "code": "not_found"}},
                status_code=404,
            )
        return sample

    @server.delete("/samples/{sample_id}")
    async def delete_sample(sample_id: int):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        deleted = samples_service.delete_sample(sample_id)
        if not deleted:
            return JSONResponse(
                {"error": {"message": "sample not found", "code": "not_found"}},
                status_code=404,
            )
        return Response(status_code=204)

    @server.get("/samples/{sample_id}/source")
    async def get_sample_source(sample_id: int):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        result = samples_service.get_sample_source(sample_id)
        if result is None:
            return JSONResponse(
                {"error": {"message": "source image not found", "code": "not_found"}},
                status_code=404,
            )
        path, mimetype = result
        return FileResponse(path, media_type=mimetype)

    @server.get("/samples/{sample_id}/blob/{kind}")
    async def get_sample_blob(sample_id: int, kind: str):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        try:
            blob = samples_service.get_sample_blob(sample_id, kind)
        except SamplesError as err:
            return _samples_error(err)
        if blob is None:
            return JSONResponse(
                {"error": {"message": "blob not present", "code": "not_found"}},
                status_code=404,
            )
        return Response(content=blob, media_type="image/png")

    @server.put("/samples/{sample_id}/blob/{kind}")
    async def put_sample_blob(sample_id: int, kind: str, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        body = await request.body() or b""
        if not body:
            return JSONResponse(
                {"error": {"message": "request body is empty", "code": "missing_blob"}},
                status_code=400,
            )
        width = request.query_params.get("width")
        height = request.query_params.get("height")
        try:
            sample = samples_service.set_sample_blob(
                sample_id, kind, body,
                width=int(width) if width else None,
                height=int(height) if height else None,
            )
        except SamplesError as err:
            return _samples_error(err)
        if sample is None:
            return JSONResponse(
                {"error": {"message": "sample not found", "code": "not_found"}},
                status_code=404,
            )
        return sample

    @server.delete("/samples/{sample_id}/blob/{kind}")
    async def delete_sample_blob(sample_id: int, kind: str):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        try:
            sample = samples_service.clear_sample_blob(sample_id, kind)
        except SamplesError as err:
            return _samples_error(err)
        if sample is None:
            return JSONResponse(
                {"error": {"message": "sample not found", "code": "not_found"}},
                status_code=404,
            )
        return sample

    # ---------------- instances ----------------

    @server.get("/samples/{sample_id}/instances")
    async def list_instances(sample_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if samples_service.get_sample(sample_id) is None:
            return JSONResponse(
                {"error": {"message": "sample not found", "code": "not_found"}},
                status_code=404,
            )
        try:
            instances = samples_service.list_instances(
                sample_id,
                include_blobs=_parse_csv_query(request.query_params.get("include_blobs")),
            )
        except SamplesError as err:
            return _samples_error(err)
        return {"instances": instances}

    @server.post("/samples/{sample_id}/instances")
    async def create_instance(sample_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not _is_json(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be application/json",
                        "code": "invalid_content_type",
                    }
                },
                status_code=400,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        try:
            instance = samples_service.create_instance(sample_id, body or {})
        except SamplesError as err:
            return _samples_error(err)
        return JSONResponse(instance, status_code=201)

    @server.put("/samples/{sample_id}/instances")
    async def replace_instances(sample_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not _is_json(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be application/json",
                        "code": "invalid_content_type",
                    }
                },
                status_code=400,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        body = body or {}
        payloads = body.get("instances")
        if not isinstance(payloads, list):
            return JSONResponse(
                {
                    "error": {
                        "message": "body must be {\"instances\": [...]}",
                        "code": "invalid_payload",
                    }
                },
                status_code=400,
            )
        try:
            instances = samples_service.replace_instances(sample_id, payloads)
        except SamplesError as err:
            return _samples_error(err)
        return {"instances": instances}

    @server.get("/instances/{instance_id}")
    async def get_instance(instance_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        try:
            instance = samples_service.get_instance(
                instance_id,
                include_blobs=_parse_csv_query(request.query_params.get("include_blobs")),
            )
        except SamplesError as err:
            return _samples_error(err)
        if instance is None:
            return JSONResponse(
                {"error": {"message": "instance not found", "code": "not_found"}},
                status_code=404,
            )
        return instance

    @server.patch("/instances/{instance_id}")
    async def patch_instance(instance_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not _is_json(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be application/json",
                        "code": "invalid_content_type",
                    }
                },
                status_code=400,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        try:
            instance = samples_service.update_instance(instance_id, body or {})
        except SamplesError as err:
            return _samples_error(err)
        if instance is None:
            return JSONResponse(
                {"error": {"message": "instance not found", "code": "not_found"}},
                status_code=404,
            )
        return instance

    @server.delete("/instances/{instance_id}")
    async def delete_instance(instance_id: int):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not samples_service.delete_instance(instance_id):
            return JSONResponse(
                {"error": {"message": "instance not found", "code": "not_found"}},
                status_code=404,
            )
        return Response(status_code=204)

    @server.get("/instances/{instance_id}/blob/{kind}")
    async def get_instance_blob(instance_id: int, kind: str):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        try:
            blob = samples_service.get_instance_blob(instance_id, kind)
        except SamplesError as err:
            return _samples_error(err)
        if blob is None:
            return JSONResponse(
                {"error": {"message": "blob not present", "code": "not_found"}},
                status_code=404,
            )
        return Response(content=blob, media_type="image/png")

    # ---------------- chat (per instance) ----------------

    @server.get("/instances/{instance_id}/chat")
    async def list_instance_chat(instance_id: int):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if samples_service.get_instance(instance_id) is None:
            return JSONResponse(
                {"error": {"message": "instance not found", "code": "not_found"}},
                status_code=404,
            )
        return {"messages": samples_service.list_chat(instance_id)}

    @server.post("/instances/{instance_id}/chat")
    async def append_instance_chat(instance_id: int, request: Request):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if not _is_json(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be application/json",
                        "code": "invalid_content_type",
                    }
                },
                status_code=400,
            )
        try:
            body = await request.json()
        except json.JSONDecodeError:
            body = {}
        body = body or {}
        try:
            message = samples_service.append_chat(
                instance_id,
                role=body.get("role", "user"),
                content=body.get("content", ""),
            )
        except SamplesError as err:
            return _samples_error(err)
        return JSONResponse(message, status_code=201)

    @server.delete("/instances/{instance_id}/chat")
    async def clear_instance_chat(instance_id: int):
        guard = _samples_or_503(samples_service)
        if guard:
            return guard
        if samples_service.get_instance(instance_id) is None:
            return JSONResponse(
                {"error": {"message": "instance not found", "code": "not_found"}},
                status_code=404,
            )
        deleted = samples_service.clear_chat(instance_id)
        return {"deleted": deleted}

    # ---------------- OpenAI-compatible chat completions ----------------

    @server.get("/v1/models")
    async def list_models():
        if chatbot is None:
            return JSONResponse(
                {
                    "error": {
                        "message": "chatbot is not configured on this server",
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=503,
            )
        return chatbot.list_models()

    @server.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        if chatbot is None:
            return JSONResponse(
                {
                    "error": {
                        "message": "chatbot is not configured on this server",
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=503,
            )

        if not _is_json(request):
            return JSONResponse(
                {
                    "error": {
                        "message": "Content-Type must be application/json",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=400,
            )

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        payload = payload or {}

        if payload.get("stream"):
            # Streaming SSE is not implemented; return a proper OpenAI error so
            # clients can fall back to non-streaming instead of hanging.
            return JSONResponse(
                {
                    "error": {
                        "message": "streaming is not supported by this server; set stream=false",
                        "type": "invalid_request_error",
                        "param": "stream",
                        "code": None,
                    }
                },
                status_code=400,
            )

        try:
            result = chatbot.chat_completion(payload)
        except ChatbotError as err:
            return JSONResponse(err.to_payload(), status_code=err.status_code)
        except Exception as err:  # noqa: BLE001
            return JSONResponse(
                {
                    "error": {
                        "message": f"internal error during chat completion: {err}",
                        "type": "server_error",
                        "param": None,
                        "code": None,
                    }
                },
                status_code=500,
            )
        return result

    return server
