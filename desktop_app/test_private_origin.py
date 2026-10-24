"""Safe ASGI check of the desktop Dropbox origin gate; never opens a picker."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import httpx


async def main():
    app_root = Path(sys.argv[1]).resolve() / "app"
    sys.path.insert(0, str(app_root))
    from spacescan_trees_demo_v1 import server

    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 53123))
    with patch.object(server.local_dropbox, "choose_images", return_value={"cancelled": True, "files": []}):
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost:18340") as client:
            allowed = await client.post("/api/dropbox-local/select", headers={"Origin": "http://localhost:18340"})
            denied = await client.post("/api/dropbox-local/select", headers={"Origin": "https://example.com"})
    if allowed.status_code != 200 or denied.status_code != 403:
        raise RuntimeError(f"Dropbox origin gate failed: allow={allowed.status_code} deny={denied.status_code}")
    print("DROPBOX_ORIGIN_OK allow=200 deny=403 picker_mocked=True")


if __name__ == "__main__":
    asyncio.run(main())
