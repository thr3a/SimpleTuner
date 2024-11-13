# SimpleTuner のための Docker

この Docker 設定は、Runpod、Vast.ai、その他の Docker 対応ホストを含むさまざまなプラットフォームで SimpleTuner アプリケーションを実行するための包括的な環境を提供します。使いやすさと堅牢性を最適化しており、機械学習プロジェクトに必要なツールやライブラリを統合しています。

## コンテナの特徴

- **CUDA 対応ベースイメージ**: GPU 加速アプリケーションをサポートするために `nvidia/cuda:11.8.0-runtime-ubuntu22.04` から構築されています。
- **開発ツール**: Git、SSH、`tmux`、`vim`、`htop` などのさまざまなユーティリティが含まれています。
- **Python とライブラリ**: Python 3.10 と、Python パッケージ管理のための必須ライブラリ `poetry` が付属しています。
- **Huggingface と WandB の統合**: Huggingface Hub と WandB とのシームレスな統合のために事前設定されており、モデルの共有や実験の追跡を容易にします。

## はじめに

### WSL を介した Windows OS サポート（実験的）

以下のガイドは、Dockerengine がインストールされた WSL2 ディストリビューションでテストされました。

### 1. コンテナのビルド

リポジトリをクローンし、Dockerfile が含まれているディレクトリに移動します。次のコマンドを使用して Docker イメージをビルドします。

```bash
docker build -t simpletuner .
```

### 2. コンテナの実行

GPU サポートでコンテナを実行するには、次のコマンドを実行します。

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

このコマンドは、GPU アクセスを持つコンテナをセットアップし、外部接続のために SSH ポートをマッピングします。

### 3. 環境変数

外部ツールとの統合を容易にするために、コンテナは Huggingface と WandB トークンのための環境変数をサポートしています。次のように実行時にこれらを渡します。

```bash
docker run --gpus all -e HUGGING_FACE_HUB_TOKEN='your_token' -e WANDB_TOKEN='your_token' -it -p 22:22 simpletuner
```

### 4. データボリューム

ホストとコンテナ間での永続的なストレージとデータ共有のために、データボリュームをマウントします。

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. SSH アクセス

コンテナへの SSH はデフォルトで設定されています。適切な環境変数を通じて SSH 公開鍵を提供することを確認してください（Vast.ai では `SSH_PUBLIC_KEY`、Runpod では `PUBLIC_KEY`）。

### 6. SimpleTuner の使用

SimpleTuner ディレクトリに移動し、Python 仮想環境をアクティブにして、アプリケーションの使用または開発を開始します。

```bash
cd SimpleTuner
source .venv/bin/activate
```

トレーニングスクリプトやその他の提供されたユーティリティをこの環境内で直接実行します。

## 追加設定

### カスタムスクリプトと設定

カスタムスタートアップスクリプトを追加したり、設定を変更したりしたい場合は、エントリスクリプト（`docker-start.sh`）を拡張して特定のニーズに合わせてください。

この設定で実現できない機能がある場合は、新しい問題をオープンしてください。

### Docker Compose

`docker-compose.yaml` を好むユーザーのために、このテンプレートが提供されており、ニーズに合わせて拡張およびカスタマイズできます。

スタックがデプロイされたら、コンテナに接続して、上記の手順に従って操作を開始できます。

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/SimpleTuner/config"
    environment:
      HUGGING_FACE_HUB_TOKEN: [your hugging face token]
      WANDB_TOKEN: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ WandB と Hugging Face のトークンを取り扱う際は注意してください！漏洩を防ぐために、プライベートなバージョン管理リポジトリにコミットしないことをお勧めします。プロダクションの使用ケースでは、キー管理ストレージを推奨しますが、これはこのガイドの範囲外です。
---

## トラブルシューティング

### CUDA バージョンの不一致

**症状**: アプリケーションが GPU を利用できない、または GPU 加速タスクを実行しようとしたときに CUDA ライブラリに関連するエラーが表示される。

**原因**: この問題は、Docker コンテナ内にインストールされている CUDA バージョンがホストマシンで利用可能な CUDA ドライバーバージョンと一致しない場合に発生する可能性があります。

**解決策**:
1. **ホストの CUDA ドライバーバージョンを確認**: ホストマシンにインストールされている CUDA ドライバーバージョンを確認するには、次のコマンドを実行します。
   ```bash
   nvidia-smi
   ```
   このコマンドは、出力の右上に CUDA バージョンを表示します。

2. **コンテナの CUDA バージョンを一致させる**: Docker イメージ内の CUDA ツールキットのバージョンがホストの CUDA ドライバーと互換性があることを確認してください。NVIDIA は一般的に前方互換性を許可していますが、NVIDIA のウェブサイトで特定の互換性マトリックスを確認してください。

3. **イメージの再構築**: 必要に応じて、Dockerfile のベースイメージをホストの CUDA ドライバーに合わせて変更します。たとえば、ホストが CUDA 11.2 を実行していて、コンテナが CUDA 11.8 用に設定されている場合、適切なベースイメージに切り替える必要があります。
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   Dockerfile を変更した後、Docker イメージを再構築します。

### SSH 接続の問題

**症状**: SSH 経由でコンテナに接続できない。

**原因**: SSH キーの設定ミスまたは SSH サービスが正しく起動していない。

**解決策**:
1. **SSH 設定の確認**: 公開 SSH キーがコンテナ内の `~/.ssh/authorized_keys` に正しく追加されていることを確認します。また、コンテナに入って次のコマンドを実行し、SSH サービスが稼働していることを確認します。
   ```bash
   service ssh status
   ```
2. **公開ポート**: コンテナを起動する際に、SSH ポート (22) が正しく公開され、マッピングされていることを確認します。実行手順は次の通りです。
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### 一般的なアドバイス

- **ログと出力**: コンテナのログや出力を確認し、問題に関するエラーメッセージや警告がないかをチェックします。
- **ドキュメントとフォーラム**: より詳細なトラブルシューティングのアドバイスについては、Docker および NVIDIA CUDA のドキュメントを参照してください。また、使用している特定のソフトウェアや依存関係に関連するコミュニティフォーラムや問題追跡システムも貴重なリソースとなります。
