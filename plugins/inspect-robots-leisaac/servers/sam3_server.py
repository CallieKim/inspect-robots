"""SAM3 text-prompt segmentation server, speaking the CaP-X ``/segment`` protocol.

Run it in an environment with ``sam3`` installed and a Hugging Face login that has access to the
gated ``facebook/sam3`` weights (not the Isaac Lab or Pyroki environments)::

    python sam3_server.py --port 8114
    python sam3_server.py --selftest path/to/frame.png "red cube"

``POST /segment`` takes ``{"image_base64": <PNG>, "text_prompt": str}`` and returns
``{"results": [{"mask_base64", "shape", "box", "score", "label"}, ...]}`` sorted by score, where
``mask_base64`` is the base64 of an ``H*W`` uint8 array (1 inside the mask). ``GET /`` reports
status.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

log = logging.getLogger("sam3_server")


class Segmenter:
    """Owns the SAM3 model and serialises requests (the processor keeps per-image state)."""

    def __init__(self, device: str = "cuda") -> None:
        started = time.perf_counter()
        self._device = device
        torch.set_default_device(device)  # SAM3 builds some tensors on the default device
        self._processor = Sam3Processor(build_sam3_image_model(device=device), device=device)
        self._lock = threading.Lock()
        log.info("SAM3 loaded on %s in %.1fs", device, time.perf_counter() - started)
        if device == "cuda":
            log.info("%.2f GiB on the GPU", torch.cuda.memory_allocated() / 2**30)

    def segment(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
        """Return one result per match, best first."""
        # The model casts activations to bfloat16 internally, so autocast is needed either way.
        autocast = torch.autocast(self._device, dtype=torch.bfloat16)
        with self._lock, torch.inference_mode(), autocast:
            state = self._processor.set_image(image)
            output = self._processor.set_text_prompt(state=state, prompt=prompt)
        masks = output["masks"].squeeze(1).to(torch.uint8).cpu().numpy()
        boxes = output["boxes"].float().cpu().numpy()
        scores = output["scores"].float().cpu().numpy()
        # Inference buffers push the resident footprint from 3.3 to 5.4 GiB. Release them so a
        # simulator sharing a 12 GB GPU is not starved.
        del output, state
        if self._device == "cuda":
            torch.cuda.empty_cache()
        results = []
        for index in np.argsort(-scores):
            mask = np.ascontiguousarray(masks[index])
            results.append(
                {
                    "mask_base64": base64.b64encode(mask.tobytes()).decode("ascii"),
                    "shape": [int(mask.shape[0]), int(mask.shape[1])],
                    "box": [float(v) for v in boxes[index]],
                    "score": float(scores[index]),
                    "label": prompt,
                }
            )
        return results


def make_handler(segmenter: Segmenter) -> type[BaseHTTPRequestHandler]:
    """Build the request handler class that serves ``segmenter`` over HTTP."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._reply(200, {"status": "ok", "model": "facebook/sam3"})

        def do_POST(self) -> None:
            if self.path != "/segment":
                self._reply(404, {"error": f"unknown path {self.path}"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                image = Image.open(io.BytesIO(base64.b64decode(body["image_base64"]))).convert(
                    "RGB"
                )
                prompt = str(body["text_prompt"])
                started = time.perf_counter()
                results = segmenter.segment(image, prompt)
                log.info(
                    "segment %r: %d result(s) in %.2fs",
                    prompt,
                    len(results),
                    time.perf_counter() - started,
                )
                self._reply(200, {"results": results})
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self._reply(400, {"error": str(exc)})
            except Exception as exc:  # report model failures instead of dropping the connection
                log.exception("segment failed")
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug(fmt, *args)

    return Handler


def selftest(segmenter: Segmenter, image_path: Path, prompt: str) -> int:
    """Segment one image and print the matches; exit code 1 when nothing matches."""
    results = segmenter.segment(Image.open(image_path).convert("RGB"), prompt)
    for result in results:
        mask = np.frombuffer(base64.b64decode(result["mask_base64"]), dtype=np.uint8)
        print(
            f"{prompt!r}: score {result['score']:.2f}, box {np.round(result['box']).tolist()}, "
            f"{int(mask.sum())} mask pixels, shape {result['shape']}"
        )
    print("SELFTEST", "PASS" if results else "FAIL")
    return 0 if results else 1


def main() -> int:
    """Parse arguments, then run the self-test or serve forever."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="cpu keeps the GPU free for a simulator on a small card, at several seconds per image",
    )
    parser.add_argument("--selftest", nargs=2, metavar=("IMAGE", "PROMPT"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    segmenter = Segmenter(args.device)
    if args.selftest:
        return selftest(segmenter, Path(args.selftest[0]), args.selftest[1])
    server = ThreadingHTTPServer((args.host, args.port), make_handler(segmenter))
    log.info("SAM3 server on http://%s:%d", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
