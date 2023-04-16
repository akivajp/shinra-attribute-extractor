# Shinra Attribute Extractor

- 森羅プロジェクト (Shinra Project) の属性抽出ツールです
  - 事前定義された属性名とWikipediaのラベル付きデータを元にしてモデルを訓練し、ページ内容(HTMLのテキスト)とページ分類(カテゴリ)情報の入力から、属性名・属性値・出現位置を抽出します
- PyTorchとHugging Face Transformers実装による事前学習モデルを使用して訓練・推論を行います
- Python 3.10.1 で動作確認を行っている最中です
