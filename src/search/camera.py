import pyglet
from search import ENABLE_VISUALIZATION

# MODIFIED FROM: https://github.com/pyglet/pyglet/blob/master/examples/window/camera.py

"""Camera class for easy scrolling and zooming.

A simple example of a Camera class that can be used to easily scroll and
zoom when rendering. For example, you might have a playfield that needs  
to scroll and/or zoom, and a GUI layer that will remain static. For that
scenario, you can create two Camera instances. You can optionally set
the minimum allowed zoom, maximum allowed zoom, and scrolling speed::

    world_camera = Camera(scroll_speed=5, min_zoom=1, max_zoom=4)
    gui_camera = Camera()

After creating Camera instances, the zoom can be easily updated. It will
clamp to the `max_zoom` parameter (default of 4)::

    world_camera.zoom += 1

The scrolling can be set in two different ways. Directly with the
`Camera.position attribute, which can be set with a tuple of absolute
x, y values::

    world_camera.position = 50, 0

Or, it can be updated incrementally with the `Camera.move(x, y)` method.
This will update the camera position by multiplying the passed vector by
the `Camera.scroll_speed` parameter, which can be set on instantiation. 

    world_camera.move(1, 0)
    # If the world_camera.scroll_speed is "5", this will move the camera
    # by 5 pixels right on the x axis. 


During your `Window.on_draw` event, you can set the Camera, and draw the
appropriate objects. For convenience, the Camera class can act as a context
manager, allowing easy use of "with"::

    @window.event
    def on_draw():
        window.clear()
    
        # Draw your world scene using the world camera
        with world_camera:
            batch.draw()
    
        # Can also be written as:
        # camera.begin()
        # batch.draw()
        # camera.end()
    
        # Draw your GUI elements with the GUI camera.
        with gui_camera:
            label.draw()

"""

if ENABLE_VISUALIZATION:

    class Camera:
        """A simple 2D camera."""

        def __init__(
            self,
            x: float = 0,
            y: float = 0,
            zoom: float = 1,
        ):
            self._window = None
            self.x = x
            self.y = y
            self.zoom = zoom

        def _begin(self):
            # Set the current camera offset so you can draw your scene.

            # Translate using the offset.
            view_matrix = self._window.view.translate(
                (-self.x * self.zoom, -self.y * self.zoom, 0)
            )
            # Scale by zoom level.
            view_matrix = view_matrix.scale((self.zoom, self.zoom, 1))

            self._window.view = view_matrix

        def _end(self):
            # Since this is a matrix, you will need to reverse the translate after rendering otherwise
            # it will multiply the current offset every draw update pushing it further and further away.

            # Reverse scale, since that was the last transform.
            view_matrix = self._window.view.scale((1 / self.zoom, 1 / self.zoom, 1))
            # Reverse translate.
            view_matrix = view_matrix.translate(
                (self.x * self.zoom, self.y * self.zoom, 0)
            )

            self._window.view = view_matrix

        def __call__(self, window: pyglet.window.BaseWindow):
            self._window = window
            return self

        def __enter__(self):
            assert (
                self._window is not None
            ), "No active window. Call with window first: `with camera(window):`"
            self._begin()

        def __exit__(self, exception_type, exception_value, traceback):
            self._end()
            self._window = None

    class CenteredCamera(Camera):
        """A simple 2D camera class. 0, 0 will be the center of the screen, as opposed to the bottom left."""

        def begin(self):
            x = -self._window.width // 2 / self.zoom + self.x
            y = -self._window.height // 2 / self.zoom + self.y

            view_matrix = self._window.view.translate(
                (-x * self.zoom, -y * self.zoom, 0)
            )
            view_matrix = view_matrix.scale((self.zoom, self.zoom, 1))
            self._window.view = view_matrix

        def end(self):
            x = -self._window.width // 2 / self.zoom + self.x
            y = -self._window.height // 2 / self.zoom + self.y

            view_matrix = self._window.view.scale((1 / self.zoom, 1 / self.zoom, 1))
            view_matrix = view_matrix.translate((x * self.zoom, y * self.zoom, 0))
            self._window.view = view_matrix
