# Copyright 2018 Mycroft AI Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit

import ctypes
import numpy as np
import random
import sdl2.ext as sdl2ext
from ctypes import c_int
from os.path import splitext
from sdl2 import (
    SDL_WINDOWEVENT_CLOSE, SDL_GetWindowID, SDL_Rect, SDL_MapRGB, SDL_FillRect, SDL_FillRects,
    SDL_WINDOWEVENT, SDL_WINDOW_RESIZABLE, SDL_WINDOWEVENT_RESIZED, SDL_HideWindow,
    SDL_GetWindowFlags, SDL_WINDOW_HIDDEN, SDL_GetCurrentDisplayMode, SDL_DisplayMode,
    SDL_GetWindowPosition, SDL_SetWindowSize, SDL_SetWindowTitle
)
from threading import Thread, Timer, Lock


class Arriz(object):
    """Array visualization window"""
    has_init = False
    windows = {}
    closed_windows = set()

    @classmethod
    def show(cls, title, data, initial_width=None, num_colors=16, grid_px=0, key=None, bg=True):
        """
        Create a new window or update an existing one

        Args:
            title (str): Title of window
            data (np.ndarray): Array to show on window
            initial_width (float): Initial width of window
            num_colors (int): How many colors of the gradient to quantize to
            grid_px (int): How many pixels of space between data points
            key (str): Unique identifier for this window. Default key is (title, data.shape)
            bg (bool): Execute without blocking in thread
        """
        key = key or title
        if key in cls.closed_windows:
            return False
        if key not in cls.windows:
            window = Arriz(title, data.shape, initial_width, num_colors, grid_px)
            cls.windows[key] = cls.windows.pop(window)
        return cls.windows[key].update(data, title=title, bg=bg)

    def __init__(self, title, data_size, initial_width=None, num_colors=16, grid_px=1):
        self.__init()
        self.data_sx, self.data_sy = data_size
        if initial_width:
            scaling = initial_width / self.data_sx
        else:
            scaling = 800 / max(data_size)
        size = (
            int(self.data_sx * scaling), int(self.data_sy * scaling)
        )
        self.title = title
        self.sdl_window = sdl2ext.Window(title, position=self._find_position(*size), size=size,
                                         flags=SDL_WINDOW_RESIZABLE)
        self.surface = self.sdl_window.get_surface()
        self.running = True
        self.scaling = scaling
        self.colors = {
            val: _calc_color(val / num_colors) for val in range(num_colors + 1)
        }
        self.num_colors = num_colors
        self.grid_px = grid_px
        self.last_data = np.zeros(data_size)
        self.draw_lock = Lock()
        self.offset_x = self.offset_y = 0
        self.windows[self] = self
        self.update_thread = None

    def update(self, arr=None, title=None, bg=True):
        if SDL_GetWindowFlags(self.sdl_window.window) & SDL_WINDOW_HIDDEN:
            return False
        if bg:
            if not self.update_thread or not self.update_thread.is_alive():
                self.update_thread = Thread(target=self._update, args=(arr, title), daemon=True)
                self.update_thread.start()
            return True
        else:
            return self.update(arr, title=title)

    def _update(self, arr, title):
        if title:
            SDL_SetWindowTitle(self.sdl_window.window, title.encode())
        if arr.shape != (self.data_sx, self.data_sy):
            self.data_sx, self.data_sy = arr.shape
            cur_sx, cur_sy = self.sdl_window.size
            cur_sz = max(cur_sx, cur_sy)
            new_sx = max(cur_sx, int(self.data_sx * self.scaling))
            new_sy = max(cur_sy, int(self.data_sy * self.scaling))
            new_sz = max(new_sx, new_sy)
            if new_sz > cur_sz:
                self.scaling /= new_sz / cur_sz
            fin_sx = max(cur_sx, int(self.data_sx * self.scaling))
            fin_sy = max(cur_sy, int(self.data_sy * self.scaling))

            if fin_sx != cur_sx or fin_sy != cur_sy:
                SDL_SetWindowSize(
                    self.sdl_window.window,
                    fin_sx, fin_sy
                )
            self._resize(fin_sx, fin_sy)
            self.last_data = arr
        with self.draw_lock:
            if arr is not None:
                self._draw_arr(arr)
            self.sdl_window.refresh()
        return True

    @classmethod
    def write_image(cls, filename, arr):
        """Write png or jpg image"""
        ext = splitext(filename)[1]
        if ext not in ['.png', '.jpg', '.jpeg']:
            filename += '.png'
        try:
            from PIL import Image
        except ImportError:
            raise ValueError('Install pillow to use write_png! (pip install pillow)')
        arr = np.rot90(cls._normalize_arr(arr), 1)
        px_arr = np.array([[list(_calc_color(val)) for val in row] for row in arr], 'uint8')
        im = Image.fromarray(px_arr, 'RGB')
        im.save(filename)

    @classmethod
    def __update_sdl(cls):
        events = sdl2ext.get_events()
        for event in events:
            if event.type == SDL_WINDOWEVENT:
                for key, window in list(cls.windows.items()):
                    if SDL_GetWindowID(window.sdl_window.window) == event.window.windowID:
                        if event.window.event == SDL_WINDOWEVENT_CLOSE:
                            SDL_HideWindow(window.sdl_window.window)
                            del cls.windows[key]
                            cls.closed_windows.add(key)

                        if event.window.event == SDL_WINDOWEVENT_RESIZED:
                            window._resize(event.window.data1, event.window.data2)
                            window.update(window.last_data)
                        break

    @classmethod
    def __init(cls):
        if not cls.has_init:
            cls.has_init = True
            sdl2ext.init()
            atexit.register(sdl2ext.quit)

            def update_daemon():
                cls.__update_sdl()
                t = Timer(0.01, update_daemon)
                t.daemon = True
                t.start()

            Thread(target=update_daemon, daemon=True).start()

    @classmethod
    def _find_position(cls, window_sx, window_sy):
        mode = SDL_DisplayMode()
        SDL_GetCurrentDisplayMode(0, ctypes.byref(mode))

        sx, sy = mode.w, mode.h
        best_score = 0
        best_pos = None
        for _ in range(100):
            score = 0.0
            rx = random.randint(0, sx - window_sx - 1)
            ry = random.randint(0, sy - window_sy - 1)

            for window in cls.windows.values():
                px, py = c_int(), c_int()
                SDL_GetWindowPosition(window.sdl_window.window, ctypes.byref(px), ctypes.byref(py))
                wx, wy = window.sdl_window.size
                cx, cy = px.value + wx // 2, py.value + wy // 2
                dx, dy = cx - rx, cy - ry
                score += (dx * dx + cy * cy) / len(cls.windows)
            if score >= best_score:
                best_score = score
                best_pos = (rx, ry)
        return best_pos

    def _resize(self, sx, sy):
        with self.draw_lock:
            self.surface = self.sdl_window.get_surface()
            scale_x = sx / self.data_sx
            scale_y = sy / self.data_sy
            self.scaling = min(scale_x, scale_y)
            self.offset_x = int((sx - self.scaling * self.data_sx) / 2)
            self.offset_y = int((sy - self.scaling * self.data_sy) / 2)

    @staticmethod
    def _normalize_arr(arr):
        arr = np.where((abs(arr) == float('inf')) | (arr != arr), 0, arr).astype('f')
        min_val, max_val = arr.min(), arr.max()
        divisor = max_val - min_val
        return (arr - min_val) / (1 if divisor == 0 else divisor)

    def _draw_arr(self, arr):
        if arr.size == 0:
            return
        self.last_data = arr
        arr = np.flip(self._normalize_arr(arr), -1)

        color_to_rects = {}
        for i, row in enumerate(arr):
            for j, val in enumerate(row):
                color = self.colors[int(val * self.num_colors)]
                px, py = int(i * self.scaling), int(j * self.scaling)
                rects = color_to_rects.setdefault(color, [])
                rects.append(SDL_Rect(
                    self.offset_x + px, self.offset_y + py,
                    int((i + 1) * self.scaling) - px - self.grid_px,
                    int((j + 1) * self.scaling) - py - self.grid_px
                ))

        bg_color = 0 if self.grid_px else 40
        _clear_screen(self.surface, (bg_color, bg_color, bg_color))
        for color, rects in color_to_rects.items():
            _draw_rects(self.surface, rects, color)

def _draw_rects(surface, rects, color):
    count = len(rects)
    varea = ctypes.cast((count * SDL_Rect)(*rects), ctypes.POINTER(SDL_Rect))
    SDL_FillRects(surface, varea, count, SDL_MapRGB(surface.format.contents, *color))


def _clear_screen(surface, color):
    SDL_FillRect(surface, None, SDL_MapRGB(surface.format.contents, *color))


def _calc_color(val):
    h, s, v = (0.7 + 0.4 * val) % 1.0, 1.0 - 0.6 * val, val * 1.0

    # Convert HSV to 24 bit RGB
    if s == 0.0:
        w = int(255 * v)
        return w, w, w
    i = int(h * 6.)  # XXX assume int() truncates!
    f = (h * 6.) - i
    p = int(255 * v * (1. - s))
    q = int(255 * v * (1. - s * f))
    t = int(255 * v * (1. - s * (1. - f)))
    v = int(255 * v)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
