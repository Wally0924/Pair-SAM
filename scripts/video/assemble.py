"""把各段影格依分鏡順序串成單一序列(硬連結,不複製)。"""
import os
import sys

SEG_DIR = os.path.expanduser('~/SAM_research-1/docs/video/segments')
ALL_DIR = f'{SEG_DIR}/_all'
ORDER = ['s1', 's2', 's3a', 's3b', 's4', 's5']


def main():
    if os.path.isdir(ALL_DIR):
        for f in os.listdir(ALL_DIR):
            os.remove(os.path.join(ALL_DIR, f))
    else:
        os.makedirs(ALL_DIR)

    k = 0
    for seg in ORDER:
        d = f'{SEG_DIR}/{seg}'
        names = sorted(n for n in os.listdir(d) if n.endswith('.png'))
        if not names:
            sys.exit(f'{seg} 沒有影格')
        for n in names:
            os.link(os.path.join(d, n), f'{ALL_DIR}/{k:06d}.png')
            k += 1
        print(f'{seg:5s} {len(names):5d} frames  ({len(names)/30:6.2f}s)')
    print(f'total {k} frames = {k/30:.2f}s -> {ALL_DIR}')


if __name__ == '__main__':
    main()
