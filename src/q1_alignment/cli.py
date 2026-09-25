from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import import_handoff, write_report
from .align_driver import build_word_timelines
from .audit import audit_aligned
from .demo import make_demo_features
from .forced_align import SUPPORTED_METHODS, align_transcript, load_ctc_bundle
from .manifest import build_manifest
from .mfa import import_mfa_json, prepare_mfa_corpus
from .pipeline import run_alignment
from .timeline import import_ctm, make_placeholder_timelines


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="CMU-MOSEI problem 1 alignment pipeline")
    commands = root.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--output", required=True)

    placeholder = commands.add_parser("make-placeholder-timelines")
    placeholder.add_argument("--manifest", required=True)
    placeholder.add_argument("--output", required=True)

    ctm = commands.add_parser("import-ctm")
    ctm.add_argument("--ctm", required=True)
    ctm.add_argument("--manifest", required=True)
    ctm.add_argument("--output", required=True)

    mfa_corpus = commands.add_parser("prepare-mfa-corpus")
    mfa_corpus.add_argument("--manifest", required=True)
    mfa_corpus.add_argument("--data-root", required=True)
    mfa_corpus.add_argument("--output-dir", required=True)
    mfa_corpus.add_argument("--ffmpeg", required=True)
    mfa_corpus.add_argument("--limit", type=int)

    mfa_json = commands.add_parser("import-mfa-json")
    mfa_json.add_argument("--input-dir", required=True)
    mfa_json.add_argument("--manifest", required=True)
    mfa_json.add_argument("--output", required=True)

    demo = commands.add_parser("make-demo-features")
    demo.add_argument("--manifest", required=True)
    demo.add_argument("--timelines", required=True)
    demo.add_argument("--output-dir", required=True)
    demo.add_argument("--limit", type=int, default=3)

    align = commands.add_parser("align")
    align.add_argument("--manifest", required=True)
    align.add_argument("--timelines", required=True)
    align.add_argument("--feature-dir", required=True)
    align.add_argument("--output-dir", required=True)
    align.add_argument("--max-positions", type=int, default=50)
    align.add_argument("--allow-placeholder", action="store_true")
    align.add_argument("--allow-partial", action="store_true")

    audit = commands.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--aligned-dir", required=True)
    audit.add_argument("--allow-partial", action="store_true")

    forced = commands.add_parser(
        "align-transcripts",
        help="produce real word-level timelines from the original videos",
    )
    forced.add_argument("--manifest", required=True)
    forced.add_argument("--data-root", required=True)
    forced.add_argument("--output", required=True)
    forced.add_argument("--method", default="energy", choices=list(SUPPORTED_METHODS))
    forced.add_argument("--ffmpeg", default="ffmpeg")
    forced.add_argument("--sample-rate", type=int, default=16000)
    forced.add_argument("--limit", type=int)
    forced.add_argument(
        "--report",
        help="optional CSV path capturing per-sample alignment diagnostics",
    )

    import_hop = commands.add_parser(
        "import-handoff",
        help="convert the teammate's document-format delivery into internal npz",
    )
    import_hop.add_argument("--handoff", required=True, help="q1_feature_handoff 目录")
    import_hop.add_argument("--manifest", required=True)
    import_hop.add_argument("--output-dir", required=True)
    import_hop.add_argument("--report", help="校验与导入结果 JSON")
    import_hop.add_argument("--strict", action="store_true", help="遇到错误立即中止")

    validate = commands.add_parser(
        "validate-handoff",
        help="dry-run validation of the teammate's delivery (writes nothing)",
    )
    validate.add_argument("--handoff", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--report", help="校验结果 JSON")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "build-manifest":
        result = build_manifest(args.data_root, args.output)
        print(json.dumps({"sample_count": len(result), "output": args.output}, ensure_ascii=False))
    elif args.command == "make-placeholder-timelines":
        result = make_placeholder_timelines(args.manifest, args.output)
        print(json.dumps({"timeline_count": len(result), "output": args.output}, ensure_ascii=False))
    elif args.command == "import-ctm":
        result = import_ctm(args.ctm, args.manifest, args.output)
        print(json.dumps({"timeline_count": len(result), "output": args.output}, ensure_ascii=False))
    elif args.command == "prepare-mfa-corpus":
        result = prepare_mfa_corpus(
            args.manifest, args.data_root, args.output_dir, args.ffmpeg, args.limit
        )
        print(json.dumps({"prepared_count": len(result), "output_dir": args.output_dir}, ensure_ascii=False))
    elif args.command == "import-mfa-json":
        result = import_mfa_json(args.input_dir, args.manifest, args.output)
        print(json.dumps({"timeline_count": len(result), "output": args.output}, ensure_ascii=False))
    elif args.command == "make-demo-features":
        result = make_demo_features(args.manifest, args.timelines, args.output_dir, args.limit)
        print(json.dumps({"feature_count": len(result), "output_dir": args.output_dir}, ensure_ascii=False))
    elif args.command == "align":
        result = run_alignment(
            args.manifest,
            args.timelines,
            args.feature_dir,
            args.output_dir,
            max_positions=args.max_positions,
            allow_placeholder=args.allow_placeholder,
            allow_partial=args.allow_partial,
        )
        print(json.dumps({"aligned_count": len(result), "output_dir": args.output_dir}, ensure_ascii=False))
    elif args.command == "audit":
        result = audit_aligned(args.manifest, args.aligned_dir, args.allow_partial)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["issue_count"]:
            raise SystemExit(1)
    elif args.command == "align-transcripts":
        result = build_word_timelines(
            args.manifest,
            args.data_root,
            args.output,
            method=args.method,
            ffmpeg_exe=args.ffmpeg,
            sample_rate=args.sample_rate,
            limit=args.limit,
            report_path=args.report,
        )
        usable = sum(1 for row in result if row["final_usable"])
        print(
            json.dumps(
                {
                    "timeline_count": len(result),
                    "final_usable": usable,
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
    elif args.command == "import-handoff":
        report = import_handoff(
            args.handoff, args.manifest, args.output_dir, strict=args.strict
        )
        if args.report:
            write_report(report, args.report)
        print(json.dumps(
            {
                "sample_count": report.sample_count,
                "imported": report.imported,
                "failed": report.failed,
                "ok": report.ok,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
        ))
        if not report.ok:
            raise SystemExit(1)
    elif args.command == "validate-handoff":
        report = _validate_only(args.handoff, args.manifest, args.report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["failed"]:
            raise SystemExit(1)


def _validate_only(handoff, manifest_path, report_path) -> dict:
    """Validate a handoff directory without writing any npz."""

    from .adapter import validate_sample
    from .common import read_jsonl
    from .common import safe_id as _safe
    from .timeline import tokenize_transcript

    records = {r["id"]: r for r in read_jsonl(manifest_path)}
    root = Path(handoff) / "samples"
    if not root.is_dir():
        return {"checked": 0, "failed": 1, "errors": [f"no samples/ under {handoff}"],
                "warnings": []}

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        record = records.get(directory.name)
        if record is None:
            for candidate, value in records.items():
                if _safe(candidate) == directory.name:
                    record = value
                    break
        if record is None:
            errors.append(f"{directory.name}: unknown sample id")
            continue
        issues = validate_sample(
            directory,
            record["id"],
            duration=float(record.get("duration_sec", 0.0)) or None,
            frame_count=int(record.get("frame_count", 0)) or None,
            expected_words=tokenize_transcript(record.get("text") or ""),
        )
        checked += 1
        errors.extend(f"{record['id']}: {m}" for m in issues.errors)
        warnings.extend(f"{record['id']}: {m}" for m in issues.warnings)

    result = {
        "checked": checked,
        "failed": len(errors),
        "errors": errors,
        "warnings": warnings,
    }
    if report_path:
        Path(report_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    main()
