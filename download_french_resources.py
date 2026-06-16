from pathlib import Path
from urllib.request import urlopen
import bz2
import gzip
import shutil
import ssl
import sys

try:
    import certifi
except ImportError:
    certifi = None


FASTTEXT_FR_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
WOLF_URL = "https://almanach.inria.fr/software_and_resources/downloads/wolf-1.0b4.xml.bz2"

MODEL_DIR = Path("models")
WOLF_DIR = Path("datasets/wolf")
FASTTEXT_GZ_PATH = MODEL_DIR / "cc.fr.300.vec.gz"
FASTTEXT_VEC_PATH = MODEL_DIR / "cc.fr.300.vec"
WOLF_BZ2_PATH = WOLF_DIR / "wolf-1.0b4.xml.bz2"
WOLF_XML_PATH = WOLF_DIR / "wolf-1.0b4.xml"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    WOLF_DIR.mkdir(parents=True, exist_ok=True)

    download_if_missing(FASTTEXT_FR_URL, FASTTEXT_GZ_PATH)
    extract_gzip_if_missing(FASTTEXT_GZ_PATH, FASTTEXT_VEC_PATH)

    download_if_missing(WOLF_URL, WOLF_BZ2_PATH)
    extract_bz2_if_missing(WOLF_BZ2_PATH, WOLF_XML_PATH)

    print()
    print("French resources ready:")
    print(FASTTEXT_VEC_PATH)
    print(WOLF_XML_PATH)


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


def extract_gzip_if_missing(input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        print("Already extracted:", output_path)
        return

    print("Extracting:", input_path)
    with gzip.open(input_path, "rb") as source:
        with output_path.open("wb") as target:
            shutil.copyfileobj(source, target)


def extract_bz2_if_missing(input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        print("Already extracted:", output_path)
        return

    print("Extracting:", input_path)
    with bz2.open(input_path, "rb") as source:
        with output_path.open("wb") as target:
            shutil.copyfileobj(source, target)


if __name__ == "__main__":
    main()
