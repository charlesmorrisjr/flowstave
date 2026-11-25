"""Copy an exported model into the Flowstave server.

  python install_model.py --run chopin                      # the server folder next to this one
  python install_model.py --run chopin --server D:\\flowstave\\server

Then restart the server and choose Music: Chopin (AI) in the app.
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def default_server():
    """The server folder: ../server in the combined project (or ../flowstave-server in older layouts)."""
    parent = os.path.dirname(ROOT)
    for name in ("server", "flowstave-server"):
        if os.path.exists(os.path.join(parent, name, "server.js")):
            return os.path.join(parent, name)
    return os.path.join(parent, "server")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="chopin")
    ap.add_argument("--name", default="chopin", help="model id shown to the app")
    ap.add_argument("--server", default=default_server())
    args = ap.parse_args()
    src = os.path.join(ROOT, "out", args.run, "export")
    if not os.path.exists(os.path.join(src, "model.onnx")):
        sys.exit(f"No export found in {src}. Run: python export_onnx.py --run {args.run}")
    if not os.path.exists(os.path.join(args.server, "server.js")):
        sys.exit(f"Flowstave server not found at {args.server}. Use --server to point to it.")
    dest = os.path.join(args.server, "models", args.name)
    os.makedirs(dest, exist_ok=True)
    for f in ("model.onnx", "model.json"):
        shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
    print(f"Installed to {dest}. Restart the Flowstave server, then pick Music: Chopin (AI) in the app.")


if __name__ == "__main__":
    main()
