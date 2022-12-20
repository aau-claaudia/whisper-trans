import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Iterator

import ffmpeg
import torch
import whisper
from whisper.utils import optional_int, write_srt


def file_duration(file_path: Path) -> float:
    info = ffmpeg.probe(file_path)
    return float(info["format"]["duration"])


def format_spend_time(start: int, end: int) -> str:
    spend_time = (end - start) / 1e9
    dt = datetime.utcfromtimestamp(spend_time)
    return dt.strftime("%H:%M:%S.%f")


def arguments() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", default="large", type=str, help="what model is used?"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Print info to screen"
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="File for transcribring",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("./out"),
        help="File for transcribring",
    )
    parser.add_argument(
        "-la",
        "--language",
        type=str,
        required=False,
        help="Language of audio, not set let whisper guess",
    )
    parser.add_argument(
        "-l", "--logging", action="store_true", default=False, help="create log file"
    )
    parser.add_argument(
        "--threads",
        type=optional_int,
        default=0,
        help="number of threads used by torch for CPU inference",
    )
    parser.add_argument("--prompt", type=str, default=[], nargs="+")
    return vars(parser.parse_args())


def write_csv(transcript: list[dict[str, str]], output_file: Path):
    with open(
        output_file.with_suffix(".csv"),
        "w",
        encoding="UTF-8",
    ) as csvfile:
        fieldnames: list[str] = list(transcript[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transcript)


def cli(args: dict[str, str]) -> None:
    output_dir: Path = args.get("output_dir")
    output_dir.mkdir(exist_ok=True)

    use_cuda = args.get("no_cuda", None) and torch.cuda.is_available()
    use_mps = args.get("no_mps", None) and torch.backends.mps.is_available()
    model_name = args.get("model")
    verbose = args.get("verbose")

    def print_v(text: str) -> None:
        if verbose:
            print(text)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transcribe_arguments = {"fp16": False}
    if args.get("language", None):
        transcribe_arguments["language"] = args.language

    if args.get("prompt", None):
        transcribe_arguments["prompt"] = args.prompt

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=output_dir / "transcribe.log",
        filemode="a",
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    files = {file for file in args.get("input", []) if file.exists()}

    start_time = perf_counter_ns()
    model = whisper.load_model(model_name, device=device)
    end_time = perf_counter_ns()

    logging.debug("Loading took %s", format_spend_time(start_time, end_time))

    for file in files:
        logging.debug(
            "Starting %s duration: %d seconds", file.name, file_duration(file)
        )
        print_v(
            f"{(80-len(file.name)-2)//2*'-'} {file.name} {(80-len(file.name)-2)//2*'-'}"
        )
        start_time = perf_counter_ns()

        result: dict[str, Any] = model.transcribe(
            file.resolve().as_posix(), **transcribe_arguments
        )
        output_file = (
            output_dir
            / f"{file.stem}_{args.get('model')}_{result.get('language', '--')}"
        )
        print(output_file)

        end_time = perf_counter_ns()
        print_v(result.keys())

        write_csv(result["segments"], output_file)

        with open(output_file.with_suffix(".srt"), "w", encoding="UTF-8") as srt:
            write_srt(result["segments"], srt)

        logging.debug(
            "End %s processed on %s threads in %s",
            file.name,
            threads,
            format_spend_time(start_time, end_time),
        )


if __name__ == "__main__":
    cli_arguments = arguments()
    cli(cli_arguments)
