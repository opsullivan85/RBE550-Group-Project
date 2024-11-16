from dataclasses import dataclass
from typing import Any
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from enum import Enum
from functools import partial
import time

from camera import Camera
from simulation import Drawable, SimObject

import pyglet

window_dims = (800, 600)
# window = pyglet.window.Window(*window_dims, "Pyglet")


class DisplayServer:
    class _Instructions(Enum):
        DISPLAY = 0
        """data: list[Drawable]"""
        CLEAR = 1
        """data: None"""
        CLOSE = 2
        """data: None"""
        SET_CAMERA = 3
        """data: Camera"""

    def __init__(self):
        parent_conn, child_conn = Pipe()
        self._conn = parent_conn
        target = partial(DisplayServer._process_target, conn=child_conn)
        self._process = Process(target=target)
        self._process.start()
        time.sleep(1.5)  # wait for process to start

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.close()

    def _send(self, instruction: _Instructions, data: Any = None):
        self._conn.send((instruction, data))

    def clear(self):
        self._send(instruction=DisplayServer._Instructions.CLEAR)

    def close(self):
        self._send(instruction=DisplayServer._Instructions.CLOSE)
        self._process.join()

    def display(self, frame: list[Drawable], clear_before: bool = True):
        if clear_before:
            self.clear()
        self._send(instruction=DisplayServer._Instructions.DISPLAY, data=frame)

    def set_camera(self, camera: Camera):
        self._send(instruction=DisplayServer._Instructions.SET_CAMERA, data=camera)

    @staticmethod
    def _process_target(conn: Connection):
        window = pyglet.window.Window(
            *window_dims,
            "Pyglet",
            config=pyglet.gl.Config(double_buffer=True),
            vsync=True,
        )
        current_agents: list[Drawable] = []
        camera = Camera()

        @window.event
        def on_draw():
            nonlocal current_agents, conn, camera
            with camera(window=window):
                window.clear()
                for agent in current_agents:
                    batch = pyglet.graphics.Batch()
                    primitives = agent.drawables(batch=batch)
                    batch.draw()

        def update(dt: float):
            nonlocal current_agents, conn, camera
            while conn.poll():
                instruction, data = conn.recv()
                instruction: DisplayServer._Instructions
                data: Any

                match instruction:
                    case DisplayServer._Instructions.DISPLAY:
                        data: list[Drawable]
                        current_agents += data

                    case DisplayServer._Instructions.CLEAR:
                        current_agents = []

                    case DisplayServer._Instructions.CLOSE:
                        pyglet.app.exit()

                    case DisplayServer._Instructions.SET_CAMERA:
                        data: Camera
                        camera = data

                    case _:
                        raise TypeError("Invalid instruction")

        pyglet.clock.schedule_interval(update, 1 / 60)
        pyglet.app.run()
        conn.close()


def display(
    frames: list[list[Drawable]],
    display_server: DisplayServer,
    frame_time: float = 1 / 60,
    clear_between: bool = True,
    clear_at_start: bool = False,
):
    if clear_at_start:
        display_server.clear()
    for frame in frames:
        display_server.display(frame, clear_before=clear_between)
        time.sleep(frame_time)


def view_collision_geometry(frame: list[SimObject], show=True, camera=Camera()):
    for agent in frame:
        for agent_part in agent.collision_polygons:
            plt.plot(*agent_part.exterior.xy)
    plt.xlim(camera.x, camera.x + window_dims[0] / camera.zoom)
    plt.ylim(camera.y, camera.y + window_dims[1] / camera.zoom)
    if show:
        plt.show()


@dataclass()
class GenericDrawable(Drawable):
    """Generic drawable, used to deffer creating pyglet geometry so it can be batched and serealized"""

    clss: list[object]
    """class to instansiate with drawables"""
    kwargss: list[dict]
    """arguments to use for drawable class creation"""

    def drawables(
        self, batch: pyglet.graphics.Batch
    ) -> list[pyglet.shapes.ShapeBase | pyglet.sprite.Sprite]:
        return [
            cls(**kwargs, batch=batch) for cls, kwargs in zip(self.clss, self.kwargss)
        ]
