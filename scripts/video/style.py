"""影片共用版面常數與繪圖工具。所有段落一致沿用,改這裡即全片套用。"""
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30

BG = '#0d0d0d'
PANEL = '#141414'
BORDER = '#333333'
TITLE_C = '#cccccc'
NOTE_C = '#7a7a7a'
SUB_C = '#ffffff'

# 方法配色,全片一致
COLORS = {
    'CMA': '#e8833a',
    'Refign': '#4f9dd9',
    'Pair-SAM': '#d94f4f',
    'SegFormer': '#9a9a9a',
    'neutral': '#8a8a8a',
}

TITLE_Y = 14
SUB_BOX_Y = (584, 630)          # 字幕黑底
SUB_TEXT_Y = 592
NOTE_Y = 666

_FONT_CACHE = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONT_CACHE:
        name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
        path = f'/usr/share/fonts/truetype/dejavu/{name}'
        _FONT_CACHE[key] = (ImageFont.truetype(path, size) if os.path.exists(path)
                            else ImageFont.load_default())
    return _FONT_CACHE[key]


def canvas():
    return Image.new('RGB', (W, H), BG)


def draw_title(d, text):
    d.text((12, TITLE_Y), text, fill=TITLE_C, font=font(16, True))


SUB_MAX_CHARS = 56
SUB_LINE_H = 34


def wrap(text, max_chars=SUB_MAX_CHARS):
    """字幕斷行,至多兩行(規格:單行 ≤60 字元、至多兩行)。"""
    words, lines, cur = text.split(), [], ''
    for w in words:
        cand = f'{cur} {w}'.strip()
        if len(cand) <= max_chars:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > 2:                       # 超過兩行時把尾巴併回第二行
        lines = [lines[0], ' '.join(lines[1:])]
    return lines


def draw_subtitle(d, text, alpha=1.0):
    """置中字幕,半透明黑底,自動斷行至多兩行。alpha 供淡入淡出使用。"""
    if not text or alpha <= 0.01:
        return
    f = font(24, True)
    lines = wrap(text)
    widths = [d.textlength(ln, font=f) for ln in lines]
    y1 = SUB_BOX_Y[1]
    y0 = y1 - 46 - (len(lines) - 1) * SUB_LINE_H
    d.rectangle([(W - max(widths)) / 2 - 18, y0, (W + max(widths)) / 2 + 18, y1],
                fill=(0, 0, 0, int(160 * alpha)))
    v = int(255 * alpha)
    for i, (ln, tw) in enumerate(zip(lines, widths)):
        d.text(((W - tw) / 2, y0 + 8 + i * SUB_LINE_H), ln, fill=(v, v, v), font=f)


def draw_note(d, text):
    f = font(13)
    d.text(((W - d.textlength(text, font=f)) / 2, NOTE_Y), text, fill=NOTE_C, font=f)


def draw_label(d, x, y, text, color='#ffffff', size=13):
    """畫格左上角的常駐標籤。"""
    f = font(size, True)
    d.rectangle([x, y, x + d.textlength(text, font=f) + 18, y + size + 9], fill='#000000cc')
    d.text((x + 6, y + 3), text, fill=color, font=f)


def fit(img, box_w, box_h, bg=PANEL):
    """等比縮放置中,不裁切、不變形。"""
    im = img.convert('RGB')
    s = min(box_w / im.width, box_h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    out = Image.new('RGB', (box_w, box_h), bg)
    out.paste(im, ((box_w - im.width) // 2, (box_h - im.height) // 2))
    return out


def white_card(img, box_w, box_h, pad=14):
    """白底示意圖放進白色圓角卡,避免直接貼在深色背景上。"""
    card = Image.new('RGB', (box_w, box_h), '#f7f7f7')
    inner = fit(img, box_w - 2 * pad, box_h - 2 * pad, bg='#f7f7f7')
    card.paste(inner, (pad, pad))
    return card


def ease(t):
    """0→1 的 ease-in-out,用於淡入與長條生長。"""
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def blend(a, b, t):
    return Image.blend(a, b, ease(t))


def save_seq(frames, out_dir, start=0):
    os.makedirs(out_dir, exist_ok=True)
    for i, fr in enumerate(frames, start=start):
        fr.save(f'{out_dir}/{i:05d}.png')
    return len(frames)


def fig_to_image(fig):
    """matplotlib figure → PIL Image。"""
    fig.canvas.draw()
    img = Image.frombuffer('RGBA', fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1)
    return img.convert('RGB')


def dark_axes(ax, fig):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#888888', labelsize=9)
    for s in ax.spines.values():
        s.set_color('#3a3a3a')
    ax.grid(color='#2a2a2a', lw=0.6, axis='y')
    ax.set_axisbelow(True)
