"""Loopback-only HTTP regressions; no real credentials or external servers."""

import asyncio

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest


@pytest.mark.parametrize("header", ["Authorization", "Cookie", "Proxy-Authorization"])
@pytest.mark.parametrize("cross_origin", [False, True])
def test_redirect_preserves_same_origin_but_removes_cross_origin_credentials(
    header, cross_origin
):
    async def scenario():
        async def received(request):
            return web.json_response(request.headers.getall(header, []))

        destination = web.Application()
        destination.router.add_get("/target", received)
        async with TestServer(destination, host="127.0.0.1") as target:
            source = web.Application()

            async def redirect(request):
                location = str(target.make_url("/target")) if cross_origin else "/target"
                raise web.HTTPFound(location)

            source.router.add_get("/redirect", redirect)
            source.router.add_get("/target", received)
            async with TestServer(source, host="127.0.0.1") as origin:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5), trust_env=False
                ) as client:
                    async with client.get(
                        origin.make_url("/redirect"),
                        headers=[(header, "fixture=one"), (header, "fixture=two")],
                    ) as response:
                        assert response.status == 200
                        assert await response.json() == (
                            [] if cross_origin else ["fixture=one", "fixture=two"]
                        )

    asyncio.run(scenario())
