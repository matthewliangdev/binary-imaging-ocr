"""Create synthetic printed-letter sheets to exercise the OCR notebook."""
from pathlib import Path
import pickle

import cv2
import numpy as np

LETTERS = 'admnopqruw'
CELL = 72
OUT = Path(__file__).resolve().parent / 'data' / 'demo'
rng = np.random.default_rng(42)
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX]


def draw_letter(sheet, letter, row, col):
    font = FONTS[int(rng.integers(len(FONTS)))]
    scale = float(rng.uniform(1.15, 1.6))
    thickness = int(rng.integers(2, 4))
    (w, h), baseline = cv2.getTextSize(letter, font, scale, thickness)
    x = col * CELL + (CELL - w) // 2
    y = row * CELL + (CELL + h) // 2 - baseline // 2
    cv2.putText(sheet, letter, (x, y), font, scale, 25, thickness, cv2.LINE_AA)
    # Use the center of the rendered foreground bounding box.
    patch = sheet[row*CELL:(row+1)*CELL, col*CELL:(col+1)*CELL]
    ys, xs = np.where(patch < 150)
    return (col*CELL + int((xs.min() + xs.max()) // 2),
            row*CELL + int((ys.min() + ys.max()) // 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for letter in LETTERS:
        sheet = np.full((10 * CELL, 8 * CELL), 255, dtype=np.uint8)
        for row in range(10):
            for col in range(8):
                draw_letter(sheet, letter, row, col)
        cv2.imwrite(str(OUT / f'{letter}.bmp'), sheet)

    test = np.full((10 * CELL, 7 * CELL), 255, dtype=np.uint8)
    classes, locations = [], []
    for row, letter in enumerate(LETTERS):
        for col in range(7):
            locations.append(draw_letter(test, letter, row, col))
            classes.append(letter)
    cv2.imwrite(str(OUT / 'test.bmp'), test)
    with (OUT / 'test_gt.pkl').open('wb') as f:
        pickle.dump({b'classes': classes, b'locations': locations}, f)
    print(f'Synthetic data generated in {OUT}')


if __name__ == '__main__':
    main()
