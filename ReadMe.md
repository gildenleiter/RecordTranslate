# 概要
```
マイクで話した内容を2秒おきに自動翻訳するAnglarアプリケーション
フロント：Angular
バックグランド：FastAPI
モデル：kotoba-tech/kotoba-whisper-v2.0, facebook/mbart-large-50-many-to-many-mmt
```

# Angular 環境構築手順 (record_translate_client)

## 1. 必要なソフトウェアのインストール

### Node.js のインストール

Angular 17 を使用するためには、**Node.js 18 以上**が推奨されますが、現在の環境では **Node.js 20.9.0** を使用します。

#### 1. バージョン確認

```sh
node -v
```

現在のバージョン: **v20.9.0**

### npm のバージョン確認

```sh
npm -v
```

現在のバージョン: **10.8.2**

---

## 2. Angular CLI のインストール

### 1. Angular CLI のバージョン確認

```sh
ng version
```

現在のバージョン:

```sh
@angular/cli                    17.3.8
```

### 2. Angular CLI のインストールまたは更新

```sh
npm install -g @angular/cli@17.3.8
```

---

## 3. Angular プロジェクトの作成

```sh
ng new my-angular-app --standalone
```

> `--standalone` オプションを指定することで、モジュールを使用しない新しい構成でプロジェクトを作成できます。

### 1. プロジェクトのディレクトリに移動

```sh
cd my-angular-app
```

### 2. 開発サーバーの起動

```sh
ng serve
```

ブラウザで `http://localhost:4200/` にアクセスすると、Angular アプリが表示されます。

---

## 4. Angular の設定ファイル

### `angular.json`

Angular プロジェクトのビルド設定や環境設定を管理します。

### `tsconfig.json`

TypeScript のコンパイル設定を管理します。

現在の TypeScript バージョン: **5.4.5**

### `environment.ts`

開発環境・本番環境で異なる設定を行うためのファイルです。

---

## 5. 依存関係のインストール

プロジェクトのルートディレクトリで以下を実行し、依存関係をインストールします。

```sh
npm install
```

---

## 6. 追加パッケージのインストール（必要に応じて）

### RxJS

現在のバージョン: **7.8.1**

```sh
npm install rxjs@7.8.1
```


# FAST 環境構築手順 (record_translate_background)

このドキュメントでは、Python 3.11.9でFastAPIの開発環境を構築する手順を説明します。

## 必要なソフトウェア

- Python 3.11.9
- Conda (MinicondaまたはAnaconda)

## 環境構築手順

### 1. Conda環境の作成

まず、FastAPIのプロジェクト用に新しいConda環境を作成します。ターミナルまたはコマンドプロンプトで以下のコマンドを実行してください。

```bash
conda env create -f environment.yml

conda activate <your-environment-name>

pip install fastapi uvicorn
```

### 2. サーバーの起動
uvicorn main:app --reload

# 動作確認

## 1. FastAPIの起動

### 1. record_translate_backgroundプロジェクトを開く
コマンドプロンプトでrecord_translate_backgroundフォルダに移動

### 2. サーバーの起動
uvicorn main:app --reload

## 2. Angular アプリケーションの起動

### 1. record_translate_clientプロジェクトを開く

### 2. アプリケーション起動
ng serve