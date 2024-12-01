import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
ENABLE_VISUALIZATION = False

# automatically detect whether or not
# pyglet will work
try:
    import pyglet

    # somehow just checking for this class
    # determines if pyglet will work
    pyglet.window.BaseWindow
    ENABLE_VISUALIZATION = True
except Exception:
    ...
finally:
    del pyglet


del sys, Path
