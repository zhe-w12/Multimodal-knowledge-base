"""Install the NLTK data resources required by Unstructured office parsers."""

from __future__ import annotations

import os
from pathlib import Path

import nltk


REQUIRED_RESOURCES = (
    ("punkt_tab", "tokenizers/punkt_tab/english/"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng/"),
)


def default_data_directory() -> Path:
    configured = os.environ.get("NLTK_DATA", "").strip()
    if configured:
        return Path(configured.split(os.pathsep, 1)[0]).expanduser()
    if os.name == "nt":
        app_data = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return app_data / "nltk_data"
    return Path.home() / "nltk_data"


def resource_available(resource: str, data_directory: Path) -> bool:
    try:
        nltk.data.find(resource, paths=[str(data_directory)])
        return True
    except LookupError:
        return False


def main() -> None:
    data_directory = default_data_directory()
    data_directory.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(data_directory))

    for package, resource in REQUIRED_RESOURCES:
        if resource_available(resource, data_directory):
            print(f"已存在：{package}")
            continue
        if not nltk.download(package, download_dir=str(data_directory), quiet=False):
            raise RuntimeError(f"NLTK 数据下载失败：{package}")
        if not resource_available(resource, data_directory):
            raise RuntimeError(f"NLTK 数据安装后仍不可用：{package}")
        print(f"已安装：{package}")

    print(f"NLTK 数据目录：{data_directory}")


if __name__ == "__main__":
    main()
