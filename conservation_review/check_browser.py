"""Exercise browser interaction against a temporary fixture DB, never real reviews."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from server import ReviewServer, ReviewStore
from test_server import ReviewTests

HERE=Path(__file__).resolve().parent
fixture=ReviewTests()
fixture.setUp()
try:
    rows=[dict(fixture.row,id=f'sample_{i}') for i in range(1,4)]
    rows[0]['review_note']='Source sign differs / 樹牌物種不同 <img src=x onerror=alert(1)>'
    fixture.manifest.write_text(json.dumps(rows),encoding='utf8')
    real_manifest = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else None
    store=ReviewStore(real_manifest or fixture.manifest,fixture.root/'browser.sqlite3')
    server=ReviewServer(('127.0.0.1',0),store)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    try:
        artifacts=HERE/'test_artifacts'
        artifacts.mkdir(exist_ok=True)
        env=dict(os.environ,REVIEW_TEST_URL=f'http://127.0.0.1:{server.server_port}',REVIEW_TEST_SCREENSHOT=str(artifacts/('browser-collection.png' if real_manifest else 'browser-fixture.png')),REVIEW_READ_ONLY='1' if real_manifest else '0')
        subprocess.run(['node',str(HERE/'test_browser.cjs')],env=env,check=True)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
finally:
    fixture.tearDown()
