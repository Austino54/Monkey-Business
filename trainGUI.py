"""
pygame_gui.py
─────────────
PyGame GUI for configuring a goal-seeking RL reward function.

reward(data) signature
──────────────────────
    data : np.ndarray, shape (5,)
           [goal_x, goal_y, player_x, player_y, prev_dist]
    return : float  –  rewPoints total

How it works
────────────
rewPoints starts at 0 each call. The three methods below are evaluated in
order. If the condition is met, the menu-configured point value is added.

Methods
───────
  gotCloser    – player is closer to goal than previous step     (+)
  gotFarther   – player is farther from goal than previous step  (-)
  reachedGoal  – player is within GOAL_TOLERANCE of goal         (+)

Point values are set via dropdowns in the GUI (-100 to +100).

Usage from another module
─────────────────────────
    import threading, numpy as np
    from pygame_gui import reward, start_gui

    threading.Thread(target=start_gui, daemon=True).start()

    data = np.array([goal_x, goal_y, player_x, player_y, prev_dist])
    r = reward(data)

Requirements
────────────
    pip install pygame numpy
"""

import subprocess
import threading
import json
import math
import sys
from pathlib import Path

import numpy as np
import pygame

# ══════════════════════════════════════════════════════════════════════════════
#  Module-level reward config  (written live by the GUI)
# ══════════════════════════════════════════════════════════════════════════════
GOT_CLOSER_PTS   : int   =  10
GOT_FARTHER_PTS  : int   = -10
REACHED_GOAL_PTS : int   = 100
GOAL_TOLERANCE   : float =  1.0   # Euclidean distance that counts as "reached"
DIST_BONUS_SCALE : int   =  25

# CONFIG_PATH = Path(__file__).with_name("reward_config.json")
CONFIG_PATH = Path(__file__).with_name("reward_config.json")
REWARD_ITEMS_KEY = "reward_items"

_PTS_LABELS = ["-100", "-75", "-50", "-25", "-10", "-5", "-1",
               "0",
               "+1",  "+5",  "+10", "+25", "+50", "+75", "+100"]
_PTS_VALUES = [-100,  -75,  -50,  -25,  -10,  -5,  -1,
                0,
                1,    5,    10,   25,   50,   75,   100]

DEFAULT_REWARD_CONFIG = {
    REWARD_ITEMS_KEY: [
        {
            "id": "GOT_CLOSER_PTS",
            "label": "Got Closer",
            "description": "curr_dist < prev_dist",
            "labels": _PTS_LABELS,
            "options": _PTS_VALUES,
            "value": 10,
        },
        {
            "id": "GOT_FARTHER_PTS",
            "label": "Got Farther",
            "description": "curr_dist > prev_dist",
            "labels": _PTS_LABELS,
            "options": _PTS_VALUES,
            "value": -10,
        },
        {
            "id": "REACHED_GOAL_PTS",
            "label": "Reached Goal",
            "description": "curr_dist ≤ goal tolerance",
            "labels": _PTS_LABELS,
            "options": _PTS_VALUES,
            "value": 100,
        },
        {
            "id": "DIST_BONUS_SCALE",
            "label": "Dist Bonus Scale",
            "description": "as curr_dist changes",
            "labels": _PTS_LABELS,
            "options": _PTS_VALUES,
            "value": 25,
        },
    ]
}


def load_reward_config(path: Path | str = CONFIG_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return json.loads(json.dumps(DEFAULT_REWARD_CONFIG))

    if isinstance(data, dict) and REWARD_ITEMS_KEY in data:
        return data

    # Backward compatibility for legacy flat config format.
    config = json.loads(json.dumps(DEFAULT_REWARD_CONFIG))
    if isinstance(data, dict):
        for item in config[REWARD_ITEMS_KEY]:
            if item["id"] in data:
                try:
                    item["value"] = type(item["value"])(data[item["id"]])
                except (TypeError, ValueError):
                    pass
    return config


def save_reward_config(config: dict, path: Path | str = CONFIG_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except OSError:
        pass


def _find_reward_item(config: dict, attr: str) -> dict | None:
    for item in config.get(REWARD_ITEMS_KEY, []):
        if item.get("id") == attr:
            return item
    return None


def _update_reward_config(attr: str, value) -> None:
    config = load_reward_config()
    item = _find_reward_item(config, attr)
    if item is not None:
        item["value"] = value
        save_reward_config(config)
# ══════════════════════════════════════════════════════════════════════════════
#  trainGUI.py now only manages config metadata; actual reward logic lives in GUIgame.py.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Colours
# ══════════════════════════════════════════════════════════════════════════════
BG         = ( 14,  17,  24)
PANEL      = ( 22,  28,  38)
PANEL2     = ( 28,  35,  50)
BORDER     = ( 46,  55,  72)
ACCENT     = ( 75, 155, 255)
ACCENT_DIM = ( 35,  80, 150)
POS_CLR    = ( 55, 200,  95)
NEG_CLR    = (215,  65,  65)
ZERO_CLR   = (120, 125, 150)
BTN_NORM   = ( 28,  85, 190)
BTN_HOV    = ( 50, 125, 255)
TEXT_PRI   = (222, 230, 245)
TEXT_SEC   = ( 95, 108, 130)
TEXT_ACC   = (120, 178, 255)
DD_BG      = ( 26,  33,  46)
DD_HOV     = ( 42,  52,  72)
DD_SEL_POS = ( 30, 110,  55)
DD_SEL_NEG = (120,  30,  30)
DD_SEL_ZRO = ( 48,  58, 100)


def _val_col(v):
    return POS_CLR if v > 0 else (NEG_CLR if v < 0 else ZERO_CLR)

def _sel_col(v):
    return DD_SEL_POS if v > 0 else (DD_SEL_NEG if v < 0 else DD_SEL_ZRO)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _rr(s, c, r, rad=8): pygame.draw.rect(s, c, r, border_radius=rad)
def _rb(s, c, r, w=1, rad=8): pygame.draw.rect(s, c, r, w, border_radius=rad)

def _wrap(text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for word in words:
        t = (cur + " " + word).strip()
        if font.size(t)[0] <= max_w: cur = t
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    return lines or [""]


# ══════════════════════════════════════════════════════════════════════════════
#  DropDown
# ══════════════════════════════════════════════════════════════════════════════
class DropDown:
    ROW_H = 26

    def __init__(self, rect, labels, values, idx, fl, fs):
        self.rect   = pygame.Rect(rect)
        self.labels = labels
        self.values = values
        self.idx    = idx
        self.open   = False
        self.hover  = -1
        self.fl, self.fs = fl, fs
        self.scroll = 0
        self.visible = len(labels)
        self._list  = pygame.Rect(self.rect.x, self.rect.bottom,
                                  self.rect.w, self.ROW_H * len(labels))

    def _configure_list_rect(self):
        screen_h = pygame.display.get_window_size()[1]
        total_h = len(self.labels) * self.ROW_H
        bottom_space = screen_h - self.rect.bottom - 10
        top_space = self.rect.top - 10

        if total_h <= bottom_space:
            self._list.top = self.rect.bottom
            self._list.h = total_h
        elif total_h <= top_space:
            self._list.h = total_h
            self._list.top = self.rect.top - total_h
        elif bottom_space >= top_space:
            self._list.top = self.rect.bottom
            self._list.h = max(bottom_space, self.ROW_H)
        else:
            self._list.h = max(top_space, self.ROW_H)
            self._list.top = self.rect.top - self._list.h

        self._list.x = self.rect.x
        self._list.w = self.rect.w
        self.scroll = 0
        self.visible = max(self._list.h // self.ROW_H, 1)

    @property
    def value(self): return self.values[self.idx]
    @property
    def label(self): return self.labels[self.idx]

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if self.rect.collidepoint(mx, my):
                self.open = not self.open
                if self.open:
                    self._configure_list_rect()
                self.hover = -1
                return False
            if self.open and self._list.collidepoint(mx, my):
                local_y = my - self._list.top
                i = self.scroll + (local_y // self.ROW_H)
                if 0 <= i < len(self.labels):
                    changed = i != self.idx
                    self.idx = i
                    self.open = False
                    return changed
            if self.open:
                self.open = False
                self.hover = -1
        if event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            if self.open and self._list.collidepoint(mx, my):
                local_y = my - self._list.top
                i = self.scroll + (local_y // self.ROW_H)
                self.hover = i if 0 <= i < len(self.labels) else -1
            else:
                self.hover = -1
        if event.type == pygame.MOUSEWHEEL and self.open:
            mx, my = pygame.mouse.get_pos()
            if self._list.collidepoint(mx, my):
                self.scroll = max(0, min(self.scroll - event.y,
                                         max(len(self.labels) - self.visible, 0)))
                return False
        return False

    def draw_header(self, surf):
        _rr(surf, PANEL2, self.rect, rad=6)
        _rb(surf, ACCENT if self.open else BORDER, self.rect, rad=6)
        lbl = self.fl.render(self.label, True, _val_col(self.value))
        surf.blit(lbl, (self.rect.x + 10, self.rect.centery - lbl.get_height() // 2))
        arr = self.fs.render("▲" if self.open else "▼", True, TEXT_SEC)
        surf.blit(arr, (self.rect.right - 16, self.rect.centery - arr.get_height() // 2))

    def draw_list(self, surf):
        if not self.open: return
        _rr(surf, DD_BG, self._list, rad=6)
        _rb(surf, ACCENT, self._list, rad=6)
        clip = surf.get_clip()
        surf.set_clip(self._list)
        start = self.scroll
        end = min(start + self.visible, len(self.labels))
        for i in range(start, end):
            lb = self.labels[i]
            vl = self.values[i]
            ir = pygame.Rect(self._list.x, self._list.y + (i - start) * self.ROW_H,
                             self._list.w, self.ROW_H)
            if i == self.idx:
                _rr(surf, _sel_col(vl), ir.inflate(-4, -4), rad=4)
                tc = TEXT_PRI
            elif i == self.hover:
                _rr(surf, DD_HOV, ir.inflate(-4, -4), rad=4)
                tc = TEXT_PRI
            else:
                tc = _val_col(vl)
            s = self.fl.render(lb, True, tc)
            surf.blit(s, (ir.x + 10, ir.centery - s.get_height() // 2))
        surf.set_clip(clip)


# ══════════════════════════════════════════════════════════════════════════════
#  App
# ══════════════════════════════════════════════════════════════════════════════
class App:
    W, H = 820, 600

    TERMINAL_COMMAND = "py -u GUImodelTraining.py"

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("RL Reward Config")
        self.clock = pygame.time.Clock()

        self.fT  = pygame.font.SysFont("Consolas",    18, bold=True)  # title
        self.fH  = pygame.font.SysFont("Consolas",    12, bold=True)  # headers
        self.fL  = pygame.font.SysFont("Consolas",    13)             # labels
        self.fS  = pygame.font.SysFont("Consolas",    11)             # small
        self.fM  = pygame.font.SysFont("Courier New", 12)             # mono
        self.fPT = pygame.font.SysFont("Consolas",    28, bold=True)  # big pts

        self.cmd_output = ""
        self.cmd_output_lines = []
        self.cmd_status = None
        self.cmd_process = None
        self.cmd_output_lock = threading.Lock()

        self._build()
        if not CONFIG_PATH.exists():
            save_reward_config(load_reward_config())

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        PAD   = 18
        TOP   = 50
        SPLIT = 320          # left col width
        rx    = SPLIT + PAD * 2
        RCOL  = self.W - rx - PAD

        # Left column
        self.cmd_panel  = pygame.Rect(PAD, TOP, SPLIT, 138)
        self.btn_run    = pygame.Rect(PAD, self.cmd_panel.bottom + 10, SPLIT, 42)
        self.output_box = pygame.Rect(PAD, self.btn_run.bottom + 10,
                                      SPLIT, self.H - self.btn_run.bottom - PAD - 10)

        # Right column – one card per reward method
        CARD_H  = 86
        CARD_GAP = 12
        self.cards = []
        y = TOP
        config = load_reward_config()
        for item in config.get(REWARD_ITEMS_KEY, []):
            opt_lbl = item.get("labels", _PTS_LABELS)
            opt_val = item.get("options", _PTS_VALUES)
            cfg_val = item.get("value", opt_val[0] if opt_val else 0)
            try:
                idx = opt_val.index(cfg_val)
            except ValueError:
                idx = 0
            dd = DropDown(
                rect   = (rx + 10, y + 45, RCOL - 18, 32),
                labels = opt_lbl,
                values = opt_val,
                idx    = idx,
                fl=self.fL, fs=self.fS,
            )
            self.cards.append({
                "attr": item.get("id", ""),
                "label": item.get("label", ""),
                "cond": item.get("description", ""),
                "dd": dd,
                "rect": pygame.Rect(rx, y, RCOL, CARD_H),
            })
            y += CARD_H + CARD_GAP

        # Formula summary panel fills remaining space
        self.sum_rect = pygame.Rect(rx, y, RCOL, self.H - y - PAD)

    # ── Loop ──────────────────────────────────────────────────────────────────
    def run(self):
        while True:
            self.clock.tick(60)
            for ev in pygame.event.get():
                self._handle(ev)
            self._draw()
            pygame.display.flip()

    # ── Events ────────────────────────────────────────────────────────────────
    def _handle(self, ev):
        if ev.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.btn_run.collidepoint(ev.pos):
                self._run_cmd()
        for card in self.cards:
            if card["dd"].handle(ev):
                _update_reward_config(card["attr"], card["dd"].value)

    # ── Terminal ──────────────────────────────────────────────────────────────
    def _run_cmd(self):
        if self.cmd_process is not None and self.cmd_process.poll() is None:
            return

        self.cmd_output_lines = []
        self.cmd_status = None
        self.cmd_process = subprocess.Popen(
            self.TERMINAL_COMMAND,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        threading.Thread(target=self._stream_process_output, daemon=True).start()

    def _stream_process_output(self):
        if self.cmd_process is None or self.cmd_process.stdout is None:
            return

        for line in self.cmd_process.stdout:
            with self.cmd_output_lock:
                self.cmd_output_lines.append(line.rstrip("\n"))
        self.cmd_process.wait()
        with self.cmd_output_lock:
            if not self.cmd_output_lines:
                self.cmd_output_lines.append("(no output)")
            self.cmd_status = self.cmd_process.returncode == 0

    # ── Draw ──────────────────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG)
        self._d_title()
        self._d_cmd_panel()
        self._d_run_btn()
        self._d_output()
        for card in self.cards:
            self._d_card(card)
        self._d_summary()
        for card in self.cards:          # lists float above everything
            card["dd"].draw_list(self.screen)

    def _d_title(self):
        t = self.fT.render("⬡  RL Reward Config  ·  Goal-Seeking Agent", True, TEXT_PRI)
        self.screen.blit(t, (18, 12))
        pygame.draw.line(self.screen, BORDER, (18, 40), (self.W - 18, 40), 1)

    def _d_cmd_panel(self):
        _rr(self.screen, PANEL, self.cmd_panel)
        _rb(self.screen, BORDER, self.cmd_panel)
        self.screen.blit(self.fH.render("TERMINAL COMMAND", True, TEXT_SEC),
                         (self.cmd_panel.x + 10, self.cmd_panel.y + 8))
        inner = pygame.Rect(self.cmd_panel.x + 10, self.cmd_panel.y + 28,
                            self.cmd_panel.w - 20, self.cmd_panel.h - 36)
        _rr(self.screen, BG, inner, rad=4)
        _rb(self.screen, ACCENT_DIM, inner, rad=4)
        y = inner.y + 7
        for line in _wrap(self.TERMINAL_COMMAND, self.fM, inner.w - 14):
            self.screen.blit(self.fM.render(line, True, POS_CLR), (inner.x + 7, y))
            y += 17

    def _d_run_btn(self):
        mx, my = pygame.mouse.get_pos()
        hov = self.btn_run.collidepoint(mx, my)
        _rr(self.screen, BTN_HOV if hov else BTN_NORM, self.btn_run)
        lbl = self.fH.render("▶   RUN COMMAND", True, TEXT_PRI)
        self.screen.blit(lbl, lbl.get_rect(center=self.btn_run.center))

    def _d_output(self):
        _rr(self.screen, PANEL, self.output_box)
        _rb(self.screen, BORDER, self.output_box)
        self.screen.blit(self.fH.render("OUTPUT", True, TEXT_SEC),
                         (self.output_box.x + 10, self.output_box.y + 8))
        if self.cmd_status is not None:
            ok  = self.cmd_status
            dot = self.fS.render("● OK" if ok else "● ERROR",
                                 True, POS_CLR if ok else NEG_CLR)
            self.screen.blit(dot, (self.output_box.right - 66, self.output_box.y + 10))
        inner = pygame.Rect(self.output_box.x + 10, self.output_box.y + 28,
                            self.output_box.w - 20, self.output_box.h - 38)
        _rr(self.screen, BG, inner, rad=4)
        if self.cmd_output_lines:
            with self.cmd_output_lock:
                lines = list(self.cmd_output_lines)
            finished = self.cmd_status is not None
            col = POS_CLR if finished and self.cmd_status else (NEG_CLR if finished else TEXT_SEC)
            wrapped = []
            for raw in lines:
                for wl in _wrap(raw or " ", self.fM, inner.w - 14):
                    wrapped.append(wl)
            max_lines = max((inner.h - 12) // 17, 1)
            display_lines = wrapped[-max_lines:]
            y = inner.y + 6
            for wl in display_lines:
                self.screen.blit(self.fM.render(wl, True, col), (inner.x + 7, y))
                y += 17
        elif self.cmd_process is not None and self.cmd_process.poll() is None:
            self.screen.blit(self.fM.render("Running...", True, TEXT_SEC),
                             (inner.x + 7, inner.y + 6))
        else:
            self.screen.blit(self.fM.render("Press RUN COMMAND …", True, TEXT_SEC),
                             (inner.x + 7, inner.y + 6))

    def _d_card(self, card):
        rect = card["rect"]
        dd   = card["dd"]
        v    = dd.value
        vc   = _val_col(v)

        # Card bg + border
        _rr(self.screen, PANEL, rect, rad=10)
        _rb(self.screen, BORDER, rect, rad=10)

        # Left accent bar
        bar = pygame.Rect(rect.x, rect.y + 8, 4, rect.h - 16)
        pygame.draw.rect(self.screen, vc, bar, border_radius=2)

        # Method name
        self.screen.blit(
            self.fH.render(card["label"].upper(), True, TEXT_ACC),
            (rect.x + 16, rect.y + 10)
        )

        # Condition
        if card["attr"] == "DIST_BONUS_SCALE":
            self.screen.blit(
                self.fS.render("WHILE   " + card["cond"] + "    THEN  rewPoints  += val/(curr_dist+1)", True, TEXT_SEC),
                (rect.x + 16, rect.y + 30)
            )
        else:
            self.screen.blit(
                self.fS.render("IF   " + card["cond"] + "    THEN  rewPoints  +=", True, TEXT_SEC),
                (rect.x + 16, rect.y + 30)
            )

        # "THEN  rewPoints +=" label
        # self.screen.blit(
        #     self.fS.render("THEN  rewPoints  +=", True, TEXT_SEC),
        #     (rect.x + 16, rect.y + 48)
        # )

        # Big point value
        sign = "+" if v >= 0 else ""
        pts_surf = self.fPT.render(f"{sign}{v}", True, vc)
        self.screen.blit(pts_surf, (rect.right - pts_surf.get_width() - 14, rect.y + 8))

        # Dropdown header (list drawn after all cards)
        dd.draw_header(self.screen)

    def _d_summary(self):
        r = self.sum_rect
        if r.h < 24: return
        _rr(self.screen, PANEL, r, rad=10)
        _rb(self.screen, BORDER, r, rad=10)
        self.screen.blit(self.fH.render("REWARD FORMULA", True, TEXT_SEC),
                         (r.x + 12, r.y + 8))

        lines = [
            ("rewPoints  =  0", ZERO_CLR),
        ]
        for card in self.cards:
            v  = card["dd"].value
            vc = _val_col(v)
            sign = "+" if v >= 0 else ""
            # if card["attr"] == "DIST_BONUS_SCALE":
            #     lines.append(f"  {sign}{v}/(d+1)  ← {card['label']}")
            # else:
            #     lines.append(f"  {sign}{v}          ← {card['label']}")
            if card["attr"] == "DIST_BONUS_SCALE":
                lines.append((f"while {card['cond']}:   rewPoints  +=  {sign}{v}/(curr_dist+1)", vc))
            else:
                lines.append((f"if {card['cond']}:   rewPoints  +=  {sign}{v}", vc))
        lines.append(("return  rewPoints", TEXT_ACC))

        y = r.y + 26
        for text, col in lines:
            if y > r.bottom - 14: break
            self.screen.blit(self.fM.render(text, True, col), (r.x + 14, y))
            y += 17


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry points
# ══════════════════════════════════════════════════════════════════════════════
def start_gui():
    """Run the GUI – call in a daemon thread from another module."""
    App().run()


if __name__ == "__main__":
    App().run()