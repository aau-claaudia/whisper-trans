import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import ffmpeg
import torch
import whisper

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="large")
parser.add_argument(
    "--no-mps", action="store_true", default=False, help="disables macOS GPU training"
)
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="disables CUDA training"
)
parser.add_argument(
    "-i", "--input", nargs="+", type=str, default="", help="File for transcribring"
)

parser.add_argument(
    "-o", "--output_dir", type=str, default="./out", help="File for transcribring"
)
parser.add_argument(
    "-la",
    "--language",
    type=str,
    default=argparse.SUPPRESS,
    help="Language of audio, not set let whisper guess",
)
parser.add_argument(
    "-l", "--logging", action="store_true", default=False, help="create log file"
)
parser.add_argument("--prompt", type=str, default=[], nargs="+")
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

transcribe_arguments = {'fp16': False}
if "language" in args:
    transcribe_arguments["language"] = args.language

if "prompt" in args:
    transcribe_arguments["language"] = args.prompt


logging.basicConfig(
    level=logging.DEBUG,
    filename="transcribe.log",
    filemode="a",
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


files = [Path(file) for file in set(args.input) if Path(file).exists()]
output_dir = Path(args.output_dir)

if not output_dir.exists():
    try:
        output_dir.mkdir()
    except FileExistsError:
        pass


def file_duration(file_path: Path) -> float:
    info = ffmpeg.probe(file_path)
    return float(info["format"]["duration"])


def format_spend_time(start: int, end: int) -> str:
    spend_time = (end - start) / 1e9
    dt = datetime.utcfromtimestamp(spend_time)
    return dt.strftime("%H:%M:%S.%f")


start_time = perf_counter_ns()
model = whisper.load_model(args.model, device=device)
end_time = perf_counter_ns()


logging.debug(f"Loading took %s", format_spend_time(start_time, end_time)")

file = files[0]
for file in files:
    logging.debug("Starting %s duration: %d seconds", file.name, file_duration(file))
    print(f"{(80-len(file.name)-2)//2*'-'} {file.name} {(80-len(file.name)-2)//2*'-'}")
    start_time = perf_counter_ns()

    # options = whisper.DecodingOptions(language='da')

    result: dict[str, Any] = model.transcribe(
        audio=file.resolve().as_posix(), **transcribe_arguments
    )

    end_time = perf_counter_ns()

    with open(
        f"{output_dir}/{file.stem}_{args.model}.csv", "w", encoding="UTF-8"
    ) as csvfile:
        fieldnames: list[str] = list(result["segments"][0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["segments"])

    logging.debug(
        "End %s processed in %d, language used: %s",
        file.name,
        format_spend_time(start_time, end_time),
        str(model.detect_language()),
    )
    # print(f"{file} processed in {format_spend_time(start_time, end_time)}")
    print()
