"""SPAD Simulator — CLI (default) or GUI (--ui)."""

import sys

if "--ui" in sys.argv:
    from .src.ui import main as ui_main
    ui_main()
else:
    import matplotlib
    matplotlib.use("Agg")
    from .src.main import main
    main()
