from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent
FONT_DIR = Path("C:/Windows/Fonts")
INK = "#eaf3fe"
SOFT = "#acc6d4"
TEAL = "#66e4c8"
LIME = "#f2c983"

def build(locale):
    img = Image.new("RGB", (1440, 1080), "#0f2536")
    draw = ImageDraw.Draw(img)
    font = lambda size, bold=False: ImageFont.truetype(
        str(FONT_DIR / ("BIZ-UDGothicB.ttc" if bold else "BIZ-UDGothicR.ttc")), size)
    draw.text((96, 77), "Vision Model Lab", font=font(34, True), fill=TEAL)
    draw.text((96, 158), "Custom Detector", font=font(81, True), fill=INK)

    ja = locale == "ja"
    subtitle = "ローカル環境で動かす物体検出モデル" if ja else "Object detection for local inference"
    draw.text((99, 269), subtitle, font=font(38), fill=SOFT)

    left, top, right, bottom = 96, 371, 1344, 862
    draw.rounded_rectangle((left, top, right, bottom), radius=26, fill="#19394e", outline="#345b70", width=2)
    for x in range(132, 1310, 52):
        draw.line((x, 395, x, 838), fill="#20465b")
    for y in range(404, 839, 52):
        draw.line((125, y, 1317, y), fill="#20465b")
    draw.line((152, 795, 1288, 795), fill="#668292", width=3)
    draw.ellipse((385, 501, 444, 560), fill="#9cc9d2")
    draw.line((414, 567, 414, 677), fill="#9cc9d2", width=36)
    draw.line((413, 589, 356, 654), fill="#9cc9d2", width=20)
    draw.line((416, 589, 470, 642), fill="#9cc9d2", width=20)
    draw.line((415, 671, 370, 769), fill="#9cc9d2", width=23)
    draw.line((418, 669, 462, 769), fill="#9cc9d2", width=23)
    draw.rounded_rectangle((921, 570, 993, 633), radius=14, outline="#eed8a8", width=9)
    draw.rounded_rectangle((827, 612, 1088, 771), radius=22, fill="#b09467", outline="#d6b787", width=4)
    draw.line((868, 623, 868, 759), fill="#eed8a8", width=16)
    draw.line((1046, 623, 1046, 759), fill="#eed8a8", width=16)
    draw.line((875, 681, 1037, 681), fill="#997a51", width=5)

    for x1, y1, x2, y2, label, color in [
        (326, 473, 501, 786, "人物" if ja else "Person", TEAL),
        (806, 540, 1109, 785, "スーツケース" if ja else "Suitcase", LIME),
    ]:
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        label_font = font(27, True)
        label_w = draw.textlength(label, font=label_font)
        draw.rounded_rectangle((x1, y1 - 42, x1 + label_w + 24, y1 + 1), radius=6, fill=color)
        draw.text((x1 + 12, y1 - 36), label, font=label_font, fill="#132b39")

    draw.text((98, 906), "640 × 640   /   PT・ONNX" if ja else "640 × 640   /   PT and ONNX",
              font=font(36, True), fill=INK)
    draw.text((98, 977), "枠は出力形式の説明図です" if ja else "Bounding boxes are illustrative.",
              font=font(26), fill=SOFT)
    img.save(ASSETS / f"listing-{locale}.png", optimize=True)

if __name__ == "__main__":
    for language in ("ja", "en"):
        build(language)

