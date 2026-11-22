"""Stale-capture detection: 0-rect answers to forced full-frame requests.

A server whose dirty-region pipeline is stalled (e.g. a Windows UAC/secure-
desktop transition) answers a non-incremental full-frame request with a
0-rect update. Before the fix, capture() returned the unchanged buffer as
success; now it retries, then flags ``last_capture_stale``.
"""

import asyncio
import struct
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_rfc6143 import (
    FILL_BGRX,
    HEIGHT,
    WIDTH,
    MockRFBServer,
    config,
    framebuffer_update,
)

from pyvnc import Rect, VNCClient, VNCConfig
from pyvnc.pyvnc_common import PIXEL_FORMATS

PIXEL_FORMAT = PIXEL_FORMATS["bgra"]


def empty_update():
    return framebuffer_update([])


def silence():
    """Script entry that answers an update request with nothing at all."""
    return b""


def full_update():
    # RAW encoding (0): the client keeps a single streaming zlib decompressor,
    # so independent zlib streams would desync its decoder.
    pixels = bytes(bytearray(FILL_BGRX * (WIDTH * HEIGHT)))
    return framebuffer_update([(0, 0, WIDTH, HEIGHT, 0, pixels)])


class ScriptedUpdateServer(MockRFBServer):
    """A MockRFBServer that answers each update request with the next scripted message.

    When the script is exhausted, the last entry is repeated. ``requests``
    counts received FramebufferUpdateRequest messages.
    """

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self.script = list(script)
        self._index = 0
        self.requests = 0
        self.clipboard_texts: list[str] = []

    async def _serve_messages(self, reader, writer):
        while True:
            message_type = (await reader.readexactly(1))[0]
            if message_type == 0:
                await reader.readexactly(19)  # SetPixelFormat
            elif message_type == 2:
                header = await reader.readexactly(3)
                count = struct.unpack("!H", header[1:3])[0]
                await reader.readexactly(4 * count)  # SetEncodings
            elif message_type == 3:
                await reader.readexactly(9)  # FramebufferUpdateRequest payload
                self.requests += 1
                entry = self.script[min(self._index, len(self.script) - 1)]
                self._index += 1
                writer.write(entry())
                await writer.drain()
            elif message_type == 4:
                await reader.readexactly(7)  # KeyEvent
            elif message_type == 5:
                await reader.readexactly(5)  # PointerEvent
            elif message_type == 6:
                header = await reader.readexactly(7)  # padding + length
                (length,) = struct.unpack("!I", header[3:7])
                self.clipboard_texts.append(
                    (await reader.readexactly(length)).decode("utf-8")
                )
            else:
                break


class TestFramebufferUpdateHandler(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the rect-count bookkeeping in _handle_framebuffer_update."""

    def _client_with_reader(self, payload: bytes) -> VNCClient:
        client = VNCClient(VNCConfig())
        client.rect = Rect(0, 0, WIDTH, HEIGHT)
        client._pixels_rgba = np.zeros((HEIGHT, WIDTH, 4), "B")
        reader = asyncio.StreamReader()
        reader.feed_data(payload)
        reader.feed_eof()
        client._reader = reader
        return client

    async def test_zero_rect_update_sets_count_and_keeps_event(self):
        client = self._client_with_reader(b"\x00" + struct.pack("!H", 0))
        client.last_capture_stale = True  # stale from an earlier cycle
        await client._handle_framebuffer_update()
        self.assertEqual(client._last_update_rect_count, 0)
        self.assertTrue(client._capture_event.is_set())
        # A 0-rect update delivers nothing new: it must NOT clear the flag.
        self.assertTrue(client.last_capture_stale)

    async def test_nonzero_rect_update_clears_stale_flag(self):
        client = self._client_with_reader(b"\x00" + struct.pack("!H", 0))
        client.last_capture_stale = True
        # Now send a real 1-rect update (a 1x1 RAW rect at 0,0).
        payload = b"\x00" + struct.pack("!H", 1)
        payload += struct.pack("!HHHH", 0, 0, 1, 1) + struct.pack("!i", 0) + bytes(4)
        client = self._client_with_reader(payload)
        client.last_capture_stale = True
        await client._handle_framebuffer_update()
        self.assertEqual(client._last_update_rect_count, 1)
        self.assertFalse(client.last_capture_stale)


class TestCaptureRetry(unittest.IsolatedAsyncioTestCase):
    async def test_capture_retries_after_empty_update(self):
        """0-rect then 1-rect: capture() retries once and returns fresh data."""
        server = ScriptedUpdateServer([full_update, empty_update, full_update])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                requests_after_connect = server.requests
                array = await client.capture(timeout=2.0)
                # The empty update triggered exactly one extra full-frame request.
                self.assertEqual(server.requests - requests_after_connect, 2)
                self.assertEqual(array.shape, (HEIGHT, WIDTH, 4))
                self.assertFalse(client.last_capture_stale)
                self.assertEqual(client._last_update_rect_count, 1)
            finally:
                await client.close()
        finally:
            server.close()

    async def test_capture_flags_stale_when_server_never_answers(self):
        """Server fully silent (no update at all): timeout path retries and flags stale.

        Regression for the field-reported variant: _last_update_rect_count keeps
        its old >=1 value when no update ever arrives, so a count-only check
        missed this case and returned stale pixels as success.
        """
        server = ScriptedUpdateServer([full_update, silence])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                self.assertFalse(client.last_capture_stale)
                requests_before = server.requests
                array = await client.capture(timeout=2.0)
                # initial wait + 3 retries, each sending one request
                self.assertEqual(server.requests - requests_before, 4)
                self.assertEqual(array.shape, (HEIGHT, WIDTH, 4))
                self.assertTrue(client.last_capture_stale)
                # counter retains the last parsed update (the initial full frame)
                self.assertEqual(client._last_update_rect_count, 1)
            finally:
                await client.close()
        finally:
            server.close()

    async def test_capture_flags_stale_when_all_retries_empty(self):
        """0-rect on every retry: buffer returned, last_capture_stale True."""
        server = ScriptedUpdateServer([empty_update])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                # connect()'s initial capture already exhausted its retries and
                # flagged the stale state; a further capture stays stale.
                self.assertTrue(client.last_capture_stale)
                array = await client.capture(timeout=2.0)
                self.assertEqual(array.shape, (HEIGHT, WIDTH, 4))
                self.assertTrue(client.last_capture_stale)
                self.assertEqual(client._last_update_rect_count, 0)
            finally:
                await client.close()
        finally:
            server.close()

    async def test_fresh_update_clears_stale_flag(self):
        """A later real update clears the flag without a capture call."""
        server = ScriptedUpdateServer([full_update])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                client.last_capture_stale = True
                client._last_update_rect_count = 0
                # Drive a real update through the handler with a mock reader.
                payload = b"\x00" + struct.pack("!H", 1)
                payload += (
                    struct.pack("!HHHH", 0, 0, 1, 1) + struct.pack("!i", 0) + bytes(4)
                )
                reader = asyncio.StreamReader()
                reader.feed_data(payload)
                reader.feed_eof()
                client._reader = reader
                await client._handle_framebuffer_update()
                self.assertFalse(client.last_capture_stale)
            finally:
                await client.close()
        finally:
            server.close()

    async def test_server_cut_text_is_stored(self):
        """A pushed ServerCutText is decoded, stored, and timestamped."""
        server = ScriptedUpdateServer([full_update])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                self.assertIsNone(client.last_cut_text)
                text = "Hello from the guest clipboard"
                wire = (
                    b"\x03"  # ServerCutText
                    + b"\x00\x00\x00"
                    + len(text).to_bytes(4, "big")
                    + text.encode("utf-8")
                )
                reader = asyncio.StreamReader()
                reader.feed_data(wire)
                reader.feed_eof()
                client._reader = reader
                listener = asyncio.create_task(client._framebuffer_listener())
                deadline = time.monotonic() + 2.0
                while client.last_cut_text is None and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                listener.cancel()
                try:
                    await listener
                except asyncio.CancelledError:
                    pass
                self.assertEqual(client.last_cut_text, text)
                self.assertGreater(client.last_cut_text_time, 0.0)
            finally:
                await client.close()
        finally:
            server.close()

    async def test_input_timestamp_postdates_frame_timestamp(self):
        """last_input_time starts below the frame time and moves past it on input."""
        server = ScriptedUpdateServer([full_update])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                frame_time = client.last_update_time
                self.assertGreater(frame_time, 0.0)
                # connect() restores the mouse position, so input_time is set;
                # typing must push it strictly past the last frame parse.
                input_after_connect = client.last_input_time
                self.assertGreater(input_after_connect, 0.0)
                await asyncio.sleep(0.01)
                await client.press("a")
                self.assertGreater(client.last_input_time, input_after_connect)
                self.assertGreater(client.last_input_time, frame_time)
            finally:
                await client.close()
        finally:
            server.close()

    async def test_set_clipboard_sends_client_cut_text(self):
        """set_clipboard serializes ClientCutText the server can frame."""
        server = ScriptedUpdateServer([full_update])
        port = await server.start()
        try:
            client = await VNCClient.connect(config(port))
            try:
                await client.set_clipboard("DISM output, pasted verbatim")
                await client.set_clipboard("emoji ✨ utf-8 fallback")
                deadline = time.monotonic() + 2.0
                while len(server.clipboard_texts) < 2 and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                self.assertEqual(
                    server.clipboard_texts, ["DISM output, pasted verbatim", "emoji ✨ utf-8 fallback"]
                )
            finally:
                await client.close()
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
