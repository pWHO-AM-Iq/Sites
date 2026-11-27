# 640 × 640 固定入力の物体検出で、矩形を元画像へ戻すまで

固定サイズの検出器へ画像を渡すときは、画像の縦横比を保ったまま縮小し、余った部分を埋めるレターボックスを使えます。枠を元画像に重ねるには、モデルに渡したときと同じリサイズ寸法と余白を使って座標を戻します。

## 比率と余白を記録する

800 × 1,200 の縦長画像を640 × 640 に入れる場合、縮尺候補は `640 / 1200` です。この値で幅と高さを整数に丸めると427 × 640 になり、残りの横幅は213ピクセル。左の余白を `213 // 2 = 106` ピクセルとすると、右には107ピクセル残ります。前後処理でこの丸め方と寄せ方を揃えます。

```python
from math import isclose

width, height = 800, 1200
ratio = 640 / max(width, height)
new_w, new_h = round(width * ratio), round(height * ratio)
left, top = (640 - new_w) // 2, (640 - new_h) // 2
scale_x, scale_y = new_w / width, new_h / height

def restore(x, y):
    return (x - left) / scale_x, (y - top) / scale_y

x, y = restore(left + new_w / 2, top + new_h / 2)
assert isclose(x, 400) and isclose(y, 600)
```

左右の境界に `restore(left, 0)` と `restore(left + new_w, new_h)` を適用すると、元画像の `(0, 0)` と `(800, 1200)` に対応します。矩形は左上と右下の両方に変換を適用し、最後に画像範囲へ収めます。パディング上だけにある矩形は、クリップ後の幅または高さが0になるため落とせます。

## モデルを入れ替えるとき

出力に使う座標順と単位を確認します。ここで扱った Custom Detector は `xyxy` 形式で入力画像上のピクセルを返します。0〜1の正規化値や `cxcywh` 形式のモデルを使う場合は、先に入力画像上の `xyxy` へ変換します。入力側で EXIF の回転を補正した場合は、出力を重ねる画像にも同じ補正を適用します。
