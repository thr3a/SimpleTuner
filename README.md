# SimpleTuner 💹

> ⚠️ **警告**: このリポジトリ内のスクリプトは、あなたのトレーニングデータを損なう可能性があります。進める前に必ずバックアップを保持してください。

**SimpleTuner** はシンプルさを重視しており、コードが容易に理解できるように設計されています。このコードベースは共有の学術的な演習として機能し、貢献を歓迎します。

## 目次

- [デザイン哲学](#design-philosophy)
- [チュートリアル](#tutorial)
- [機能](#features)
  - [Flux](#flux1)
  - [PixArt Sigma](#pixart-sigma)
  - [Stable Diffusion 2.0/2.1](#stable-diffusion-20--21)
  - [Stable Diffusion 3.0](#stable-diffusion-3)
  - [Kwai Kolors](#kwai-kolors)
- [ハードウェア要件](#hardware-requirements)
  - [Flux](#flux1-dev-schnell)
  - [SDXL](#sdxl-1024px)
  - [Stable Diffusion (レガシー)](#stable-diffusion-2x-768px)
- [スクリプト](#scripts)
- [ツールキット](#toolkit)
- [セットアップ](#setup)
- [トラブルシューティング](#troubleshooting)

## デザイン哲学

- **シンプルさ**: ほとんどの使用ケースに対して良好なデフォルト設定を持つことを目指し、調整が少なくて済むようにしています。
- **多様性**: 小規模なデータセットから大規模なコレクションまで、幅広い画像量を扱えるように設計されています。
- **最先端の機能**: 効果が証明された機能のみを取り入れ、未検証のオプションの追加を避けています。

## チュートリアル

[チュートリアル](/TUTORIAL.md)に取り掛かる前に、このREADMEを十分に探索してください。重要な情報が含まれている可能性があります。

完全なドキュメントを読むことなくすぐに始めたい場合は、[クイックスタート](/documentation/QUICKSTART.md)ガイドを使用できます。

メモリ制約のあるシステムについては、🤗Accelerateを使用してMicrosoftのDeepSpeedを最適化状態オフロードに設定する方法を説明した[DeepSpeedドキュメント](/documentation/DEEPSPEED.md)を参照してください。

マルチノード分散トレーニングについては、[このガイド](/documentation/DISTRIBUTED.md)がINSTALLおよびクイックスタートガイドからの設定を調整し、数十億のサンプル数の画像データセットに適したマルチノードトレーニングを最適化するのに役立ちます。

---

## 機能

- マルチGPUトレーニング
- 画像とキャプションの特徴（埋め込み）が事前にハードドライブにキャッシュされるため、トレーニングがより速く、メモリ消費が少なくなります。
- アスペクトバケット: 様々な画像サイズとアスペクト比をサポートし、ワイドスクリーンおよびポートレートトレーニングを可能にします。
- SDXLのためのリファイナLoRAまたはフルu-netトレーニング
- ほとんどのモデルは24G GPUでトレーニング可能で、低いベース解像度では16Gまで対応可能です。
  - 16G VRAM未満で動作するPixArt、SDXL、SD3、およびSD 2.xのためのLoRA/LyCORISトレーニング
- DeepSpeed統合により、[12GのVRAMでSDXLのフルu-netをトレーニング](/documentation/DEEPSPEED.md)できますが、非常に遅くなります。
- 低精度ベースモデルを使用してVRAM消費を削減するための量子化NF4/INT8/FP8 LoRAトレーニング。
- モデルの過学習を抑制し、トレーニングの安定性を向上させるためのオプションのEMA（指数移動平均）重みネットワーク。**注意:** これはLoRAには適用されません。
- 高価なローカルストレージの必要がなく、S3互換のストレージプロバイダーから直接トレーニングできます。（Cloudflare R2およびWasabi S3でテスト済み）
- SDXLおよびSD 1.x/2.x専用のフル[ControlNetモデルトレーニング](/documentation/CONTROLNET.md)（ControlLoRAやControlLiteではありません）
- 軽量で高品質な拡散モデルのための[Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md)トレーニング
- 優れた収束と過学習の低減のための[マスクドロストレーニング](/documentation/DREAMBOOTH.md#masked-loss)を提供
- LyCORISモデルのための強力な[事前正則化](/documentation/DATALOADER.md#is_regularisation_data)トレーニングサポート
- トレーニングの進捗、検証、エラーをDiscordチャンネルなどに更新するためのWebhookサポート
- モデルのアップロードをシームレスに行い、自動生成されたモデルカードを提供する[Hugging Face Hub](https://huggingface.co)との統合。

### Flux.1

Flux.1のフルトレーニングサポートが含まれています：

- クラシファイアフリーガイダンストレーニング
  - 無効のままにして、開発モデルの蒸留特性を保持します。
  - または、CFGをモデルに再導入し、推論速度とトレーニング時間のコストで創造性を向上させます。
- （オプション）優れた細部と一般化能力のためのT5注意マスクトレーニング
- 単一GPUでのLoRAまたはフルチューニングをDeepSpeed ZeROを使用して実施
- 主要なメモリ節約のために、`--base_model_precision`を`int8-quanto`または`fp8-quanto`に設定してベースモデルを量子化
```

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### PixArt Sigma

SimpleTuner は PixArt Sigma との広範なトレーニング統合を提供しており、600M および 900M モデルは修正なしで読み込むことができます。

- テキストエンコーダーのトレーニングはサポートされていません。T5 は非常に大きいためです。
- LyCORIS とフルチューニングは期待通りに動作します。
- ControlNet のトレーニングはまだサポートされていません。
- [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) トレーニングサポート（参照: [MIXTURE_OF_EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md)）

トレーニングを開始するには、[PixArt Quickstart](/documentation/quickstart/SIGMA.md) ガイドを参照してください。

### Stable Diffusion 3

- LoRA とフルファインチューニングは通常通りサポートされています。
- ControlNet はまだ実装されていません。
- セグメント化されたタイムステップ選択や Compel 長いプロンプトの重み付けなどの特定の機能はまだサポートされていません。
- パラメータは最良の結果を得るために最適化されており、SD3 モデルのゼロからのトレーニングを通じて検証されています。

始めるには、[Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) を参照してください。

### Kwai Kolors

ChatGLM (General Language Model) 6B をテキストエンコーダーとして使用した SDXL ベースのモデルで、**隠れ次元のサイズを倍増**し、プロンプト埋め込みに含まれるローカル詳細のレベルを大幅に増加させています。

Kolors のサポートは、ControlNet トレーニングサポートを除いて、SDXL とほぼ同じ深さです。

### Legacy Stable Diffusion models

RunwayML の SD 1.5 と StabilityAI の SD 2.x は、どちらも `legacy` の指定の下でトレーニング可能です。

---

## Hardware Requirements

### NVIDIA

3080 以上のほぼすべてのモデルが安全な選択です。YMMV。

### AMD

LoRA とフルランクチューニングは、7900 XTX 24GB および MI300X で動作することが確認されています。

`xformers` がないため、Nvidia の同等ハードウェアよりも多くのメモリを使用します。

### Apple

LoRA とフルランクチューニングは、128G メモリを搭載した M3 Max で動作することが確認されており、SDXL には約 **12G** の「Wired」メモリと **4G** のシステムメモリを使用します。
  - M シリーズハードウェアでの機械学習には、メモリ効率の良いアテンションが不足しているため、24G 以上のマシンが必要になる可能性があります。
  - MPS に関する Pytorch の問題を購読するのは良いアイデアかもしれません。ランダムなバグがトレーニングを停止させることがあります。

### Flux.1 [dev, schnell]

- A100-80G (フルチューン、DeepSpeed 使用)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

Flux は複数の大きな GPU でのトレーニングを好みますが、単一の 16G カードでもトランスフォーマーとテキストエンコーダーの量子化を行うことで実行可能です。

### SDXL, 1024px

- A100-80G (EMA、大きなバッチ、LoRA @ 非常に大きなバッチサイズ)
- A6000-48G (EMA@768px、EMA@1024px はなし、LoRA @ 高バッチサイズ)
- A100-40G (EMA@1024px はなし、EMA@768px はなし、EMA@512px、LoRA @ 高バッチサイズ)
- 4090-24G (EMA@1024px はなし、バッチサイズ 1-4、LoRA @ 中高バッチサイズ)
- 4080-12G (LoRA @ 低中バッチサイズ)

### Stable Diffusion 2.x, 768px

- 16G 以上


## Toolkit

SimpleTuner に付属するツールキットに関する詳細は、[ツールキットのドキュメント](/toolkit/README.md)を参照してください。

## Setup

詳細なセットアップ情報は、[インストールドキュメント](/INSTALL.md)にあります。

## Troubleshooting

デバッグログを有効にするには、環境ファイル (`config/config.env`) に `export SIMPLETUNER_LOG_LEVEL=DEBUG` を追加してください。

トレーニングループのパフォーマンス分析のために、`SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` を設定すると、構成の問題を強調するタイムスタンプが表示されます。

利用可能なオプションの包括的なリストについては、[このドキュメント](/OPTIONS.md)を参照してください。

## Discord

さらにサポートが必要な場合や、同じ志を持つ人々とトレーニングについて話し合いたい場合は、[私たちの Discord サーバー](https://discord.gg/cSmvcU9Me9)に参加してください。
