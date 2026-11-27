# 横断歩道の推論作例

公式サイトの `main/dist/assets/crossing.jpg` は、内蔵 ImageGen で作成した画像をJPEG化したもの。データセット由来の画像は含まない。`make_example.py` がこのJPEGをNMSフリー版ONNXで処理し、元画像のハッシュ・検出枠・信頼度を `main/dist/data/example.json` に出力する。表示閾値0.50。画像ごとの検出例として掲載する。

## 生成に使ったプロンプト

Use case: photorealistic-natural. Make the input image for a real object detection demo for a Japanese vision research lab. Very wide landscape frame, elevated three-quarter view onto a modern Japanese street crossing. A single striking cobalt blue compact car clearly identifiable near right center, a few individual ordinary people walking and one bicycle near left, broad bold white zebra crossing stripes cut diagonally across charcoal asphalt. Strong precise afternoon shadows. Almost monochrome editorial architecture photograph with isolated electric blue paint on car, tactile grain, high contrast. Realistic physics. Frame subjects completely. No graphic overlays, no annotations, no signage text, no letters, no branding. Aim 16:9 landscape.
