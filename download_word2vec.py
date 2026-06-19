from pathlib import Path
from urllib.request import urlopen
import shutil
import ssl
import sys

try:
    import certifi
except ImportError:
    certifi = None


WORD2VEC_URL = (
    "https://media.githubusercontent.com/media/mmihaltz/"
    "word2vec-GoogleNews-vectors/master/GoogleNews-vectors-negative300.bin.gz"
)

MODEL_DIR = Path("models")
WORD2VEC_PATH = MODEL_DIR / "GoogleNews-vectors-negative300.bin.gz"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download_if_missing(WORD2VEC_URL, WORD2VEC_PATH)

    print()
    print("Word2Vec resource ready:")
    print(WORD2VEC_PATH)


def download_if_missing(url: str, output_path: Path) -> None:
    if output_path.exists():
        print("Already downloaded:", output_path)
        return

    print("Downloading:")
    print(url)
    print("to", output_path)

    if certifi is None:
        context = ssl.create_default_context()
    else:
        context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(url, context=context) as response:
        with output_path.open("wb") as handle:
            copy_with_progress(response, handle, response.headers.get("Content-Length"))
    print()


def copy_with_progress(source, target, total_size: str | None) -> None:
    total_size = int(total_size) if total_size is not None else None
    downloaded = 0
    chunk_size = 1024 * 1024

    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break

        target.write(chunk)
        downloaded += len(chunk)

        if total_size:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rProgress: {percent:6.2f}%")
        else:
            downloaded_mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloaded: {downloaded_mb:.1f} MB")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
