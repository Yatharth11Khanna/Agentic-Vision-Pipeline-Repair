"""Launch the Streamlit experiment app with the project's Python environment."""

import os
import sys


def main() -> None:
    port = os.environ.get("PORT", "5000")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            port,
            "--server.address",
            "0.0.0.0",
        ],
    )


if __name__ == "__main__":
    main()
