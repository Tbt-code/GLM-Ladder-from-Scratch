"""Tokenize one source into val/train shards and upload each to a Hugging Face dataset repo.

Resumable: a per-source manifest in the repo records finished shards and how many
source documents they consumed; a rerun skips those documents and continues.
Each shard is uploaded before the manifest that lists it, so a crash in between
only causes the same (deterministic) shard to be rebuilt and overwritten.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Callable

from .data_sources import SOURCES, iter_documents
from .shards import ShardInfo, ShardWriter, tokenize_stream
from .tokenizer import EOS_ID, load_tokenizer


def load_manifest(api, repo_id: str, source: str) -> dict:
    filename = f"{source}/manifest.json"
    if not api.file_exists(repo_id, filename, repo_type="dataset"):
        return {"source": source, "shards": []}
    path = api.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", force_download=True)
    return json.loads(Path(path).read_text())


def tokenize_source(
    api, repo_id: str, source: str, *,
    tokenizer_path: str | Path, train_tokens: int, val_tokens: int,
    shard_tokens: int, workdir: str | Path, log: Callable[[str], None] = print,
) -> dict:
    workdir = Path(workdir) / source
    tokenizer_path = Path(tokenizer_path)
    manifest = load_manifest(api, repo_id, source)
    manifest.update({
        "dataset": SOURCES[source][:2],
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "dtype": "uint16", "eos_id": EOS_ID,
    })
    shards = manifest["shards"]
    val_done = any(s["name"].startswith("val-") for s in shards)
    train_shards = [s for s in shards if s["name"].startswith("train-")]
    train_written = sum(s["tokens"] for s in train_shards)
    consumed = shards[-1]["documents_end"] if shards else 0
    if val_done and train_written >= train_tokens:
        log(f"[{source}] already complete: {train_written:,} train tokens")
        return manifest

    log(f"[{source}] resuming after {consumed:,} documents, {train_written:,} train tokens done")
    tokenizer = load_tokenizer(tokenizer_path)
    stream = tokenize_stream(tokenizer, islice(iter_documents(source), consumed, None))
    started, tokens_this_session = time.perf_counter(), 0

    def publish(info: ShardInfo) -> None:
        nonlocal tokens_this_session
        local = workdir / info.name
        api.upload_file(
            path_or_fileobj=str(local), path_in_repo=f"{source}/{info.name}",
            repo_id=repo_id, repo_type="dataset", commit_message=f"{source}: add {info.name}",
        )
        local.unlink()
        shards.append(asdict(info))
        manifest["train_tokens"] = sum(s["tokens"] for s in shards if s["name"].startswith("train-"))
        api.upload_file(
            path_or_fileobj=json.dumps(manifest, indent=2).encode(), path_in_repo=f"{source}/manifest.json",
            repo_id=repo_id, repo_type="dataset", commit_message=f"{source}: manifest after {info.name}",
        )
        tokens_this_session += info.tokens
        rate = tokens_this_session / (time.perf_counter() - started)
        log(f"[{source}] {info.name}: {info.tokens:,} tokens, {info.skipped_documents} skipped docs, "
            f"{rate:,.0f} tokens/s")

    if not val_done:
        writer = ShardWriter(workdir, "val", val_tokens, documents_consumed=consumed)
        for document in stream:
            if (info := writer.add(document)) is not None:
                publish(info)
                consumed = info.documents_end
                break

    writer = ShardWriter(workdir, "train", shard_tokens,
                         start_index=len(train_shards), documents_consumed=consumed)
    for document in stream:
        if train_written >= train_tokens:
            break
        if (info := writer.add(document)) is not None:
            publish(info)
            train_written += info.tokens
    else:  # the source ran out before the target: keep the partial last shard
        if (info := writer.flush()) is not None:
            publish(info)
            train_written += info.tokens

    log(f"[{source}] done: {train_written:,} train tokens")
    return manifest