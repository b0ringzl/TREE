"""Loopback-only Demo sandbox, pointing to the separately checked teacher."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn
from spacescan_trees_demo_v1 import server

server.UPSTREAM = 'http://127.0.0.1:8040'
if __name__ == '__main__':
    uvicorn.run(server.app, host='127.0.0.1', port=8057)
