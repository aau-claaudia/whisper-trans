#!/usr/bin/env python

import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
import whisper

from whispaau.archive import archiving
from whispaau.cli_utils import parse_arguments
from whispaau.logging import Logger
from whispaau.utils import get_writer


def cli(args: dict[str, Any]) -> None:
    job_name = args.get("job_name")
    output_dir: Path = args.get("output_dir")
    output_dir.mkdir(exist_ok=True)
    verbose = args.get("verbose")
    log = Logger(name=job_name, output_dir=output_dir, verbose=verbose)

    # Setup CPU/GPU and model
    use_cuda = not args.get("no_cuda") and torch.cuda.is_available()
    use_mps = not args.get("no_mps") and torch.backends.mps.is_available()
    model_name = args.get("model")

    secret_password = args.get("archive_password", None)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transcribe_arguments = {"fp16": False}
    if args.get("language", None):
        transcribe_arguments["language"] = args.get("language")

    if args.get("prompt", None):
        transcribe_arguments["prompt"] = args.get("prompt")

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    files = args.get("input")

    start_time = perf_counter_ns()
    model = whisper.load_model(model_name, device=device)
    log.log_model_loading(model_name, start_time, perf_counter_ns())

    output_format = args.pop("output_format")
    writer = get_writer(output_format, output_dir)

    log.log_processing(files)

    for file in files:
        process_file(
            log,
            file,
            output_dir,
            model_name,
            model,
            device,
            writer,
            transcribe_arguments,
        )

    # Scan for generated files in output_dir:
    files_to_pack = [path for path in output_dir.glob("*") if path.is_file()]
    # Pack everything into a process_name.zip
    job_name_directory = Path(job_name)
    archiving(
        jobname=job_name_directory,
        output_file=output_dir / job_name_directory.with_suffix(".zip"),
        paths=files_to_pack,
        secret_password=secret_password,
    )


def process_file(
    log: Logger,
    file: Path,
    output_dir,
    model_name,
    model,
    device,
    writer,
    trans_arguments,
) -> None:
    log.log_file_start(file, device)
    start_time = perf_counter_ns()

    result: dict[str, Any] = model.transcribe(
        file.resolve().as_posix(), **trans_arguments
    )
    output_file = (
        output_dir / f"{file.stem}_{model_name}_{result.get('language', '--')}"
    )

    writer(
        result,
        output_file,
        options={
            "highlight_words": None,
            "max_line_count": None,
            "max_line_width": None,
        },
    )

    log.log_file_end(file, start_time, perf_counter_ns())


if __name__ == "__main__":
    arguments = sys.argv[1:]
    # print(arguments)
    cli_arguments = parse_arguments(None)
    # print(cli_arguments)
    cli(cli_arguments)
