from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path

from btcore.data.cache import BarCache
from btcore.data.providers import get_provider
from btcore.data.resample import resample_1m_to_5m
from btcore.engine import run
from btcore.models import BacktestConfig
from btcore.report import write_report
from btcore.safety import validate_strategy_file, validate_strategy_source


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")

    fetch = subparsers.add_parser("fetch")
    _add_run_args(fetch)

    run_parser = subparsers.add_parser("run")
    _add_run_args(run_parser)
    run_parser.add_argument("--strategy", default="strategies/moving_average.py")
    run_parser.add_argument("--output", default="runs/latest")
    run_parser.add_argument("--fill-price", default="next_open")

    report = subparsers.add_parser("report")
    report.add_argument("--run", default="latest")

    patch = subparsers.add_parser("patch-strategy")
    patch.add_argument("--strategy", required=True)
    patch.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    patch.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "init":
        _init_project()
    elif args.command == "fetch":
        _fetch(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "report":
        _report(args)
    elif args.command == "patch-strategy":
        _patch_strategy(args)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbols", nargs="+", default=["000001.XSHE"])
    parser.add_argument("--freq", choices=["1d", "1m", "5m"], default="1d")
    parser.add_argument("--start", default="2026-01-05")
    parser.add_argument("--end", default="2026-01-05")
    parser.add_argument("--provider", default="jqdata")
    parser.add_argument("--adj", default="pre")
    parser.add_argument("--cache-dir", default="data/cache")


def _init_project() -> None:
    Path("strategies").mkdir(exist_ok=True)
    strategy = Path("strategies/moving_average.py")
    if not strategy.exists():
        strategy.write_text(_default_strategy(), encoding="utf-8")
    Path("configs").mkdir(exist_ok=True)
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    print("Initialized btcore project.")


def _fetch(args: argparse.Namespace) -> None:
    config = BacktestConfig(symbols=tuple(args.symbols), frequency=args.freq, start=args.start, end=args.end, provider=args.provider, adj=args.adj, cache_dir=args.cache_dir)
    provider = get_provider(config.provider, config.sessions)
    fetch_freq = "1m" if config.frequency == "5m" and config.resample_policy == "from_1m" else config.frequency
    bars = provider.fetch_bars(config.symbols, fetch_freq, config.start, config.end, config.adj)
    cache = BarCache(config.cache_dir)
    if config.frequency == "5m" and fetch_freq == "1m":
        cache.write(config.provider, fetch_freq, config.adj, bars)
        bars = resample_1m_to_5m(bars)
    paths = cache.write(config.provider, config.frequency, config.adj, bars)
    print(f"Wrote {len(bars)} bars into {len(paths)} cache partitions.")


def _run(args: argparse.Namespace) -> None:
    config = BacktestConfig(symbols=tuple(args.symbols), frequency=args.freq, start=args.start, end=args.end, provider=args.provider, adj=args.adj, cache_dir=args.cache_dir, fill_price=args.fill_price)
    validate_strategy_file(args.strategy, require_strategies_dir=False)
    result = run(config, args.strategy)
    out = write_report(result, args.output)
    print(json.dumps(result.metrics, indent=2))
    print(f"Report written to {out}")


def _report(args: argparse.Namespace) -> None:
    root = Path("runs") / args.run
    metrics = root / "metrics.json"
    if not metrics.exists():
        raise SystemExit(f"No report found at {root}")
    print(metrics.read_text(encoding="utf-8"))


def _patch_strategy(args: argparse.Namespace) -> None:
    validate_strategy_file(args.strategy)
    if args.set:
        path = Path(args.strategy)
        original = path.read_text(encoding="utf-8")
        updated = _update_strategy_params(original, args.set)
        validate_strategy_file(path)
        if args.dry_run:
            print("".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True), fromfile=str(path), tofile=str(path))))
            return
        path.write_text(updated, encoding="utf-8")
        validate_strategy_file(path)
        print(f"Updated {path}")
        return
    if args.dry_run:
        print("Strategy passed static safety checks. No patch was applied.")
        return
    raise SystemExit("Provide --set NAME=VALUE to modify strategy parameters, or use --dry-run for safety checks only.")


def _update_strategy_params(source: str, assignments: list[str]) -> str:
    updates: dict[str, str] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise SystemExit(f"Invalid --set value: {assignment}")
        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        if not name.isidentifier():
            raise SystemExit(f"Invalid parameter name: {name}")
        ast.literal_eval(raw_value)
        updates[name] = raw_value

    lines = source.splitlines()
    found: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                if target.attr in updates and node.lineno == node.end_lineno:
                    prefix = lines[node.lineno - 1].split("self.", 1)[0]
                    lines[node.lineno - 1] = f"{prefix}self.{target.attr} = {updates[target.attr]}"
                    found.add(target.attr)
    missing = sorted(set(updates) - found)
    if missing:
        raise SystemExit(f"Strategy parameters not found: {', '.join(missing)}")
    updated = "\n".join(lines)
    if source.endswith("\n"):
        updated += "\n"
    validate_strategy_source(updated)
    return updated


def _default_strategy() -> str:
    return '''class Strategy:
    frequency = "1m"

    def __init__(self):
        self.fast = 5
        self.slow = 20

    def on_bar(self, ctx, data):
        for symbol in data:
            fast = ctx.history(symbol, "close", self.fast)
            slow = ctx.history(symbol, "close", self.slow)
            if len(slow) < self.slow:
                continue
            if sum(fast) / len(fast) > sum(slow) / len(slow):
                ctx.order_target_percent(symbol, 0.95)
            else:
                ctx.order_target_percent(symbol, 0.0)
'''


if __name__ == "__main__":
    main()
