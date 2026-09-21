"""Synthetic, imbalanced transaction generator and high-throughput stream client.

All data is SYNTHETIC (no PII). Scenarios:
  * normal     - cards transact from a home device/IP at popular merchants
  * ring       - coordinated cards share devices/IPs and hit colluding merchants in bursts
  * ato        - account takeover: new device + new IP, high amounts, short burst

Usage:
  python -m src.api.stream_simulator --url http://localhost:8000 --n 2000 --rate 100
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import numpy as np

DAY = 86_400.0


def generate_transactions(
    n_normal: int = 30_000,
    n_cards: int = 3_000,
    n_merchants: int = 400,
    n_rings: int = 30,
    n_ato: int = 150,
    days: float = 30.0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate a chronologically ordered list of labelled synthetic transactions."""
    rng = np.random.default_rng(seed)
    home_device = rng.integers(0, int(n_cards * 1.05), n_cards)
    home_ip = rng.integers(0, int(n_cards * 0.8), n_cards)
    merchant_pop = rng.dirichlet(np.full(n_merchants, 0.3))
    rows: list[dict[str, Any]] = []

    cards = rng.integers(0, n_cards, n_normal)
    roam_dev = rng.random(n_normal) < 0.04
    roam_ip = rng.random(n_normal) < 0.10
    devices = np.where(roam_dev, rng.integers(0, int(n_cards * 1.05), n_normal), home_device[cards])
    ips = np.where(roam_ip, rng.integers(0, int(n_cards * 0.8), n_normal), home_ip[cards])
    merchants = rng.choice(n_merchants, n_normal, p=merchant_pop)
    # legitimate users also appear on brand-new devices/IPs (new phone, travel, public Wi-Fi)
    fresh_dev = rng.random(n_normal) < 0.03
    fresh_ip = rng.random(n_normal) < 0.06
    amounts = rng.lognormal(3.6, 1.0, n_normal)
    times = rng.uniform(0, days * DAY, n_normal)
    for k, (c, d, i, m, a, t) in enumerate(zip(cards, devices, ips, merchants, amounts, times)):
        dev_id = f"dev_new{k}" if fresh_dev[k] else f"dev_{d}"
        ip_id = f"ip_new{k}" if fresh_ip[k] else f"ip_{i}"
        rows.append(dict(timestamp=float(t), card_id=f"card_{c}", device_id=dev_id, ip_id=ip_id,
                         merchant_id=f"m_{m}", amount=round(float(a), 2), label=0, scenario="normal"))

    for r in range(n_rings):
        members = rng.choice(n_cards, int(rng.integers(4, 10)), replace=False)
        # rings mostly route through compromised, already-active devices/IPs (harder to spot)
        ring_devs = [f"dev_{d}" for d in rng.choice(int(n_cards * 1.05), int(rng.integers(1, 4)), replace=False)]
        ring_ips = [f"ip_{i}" for i in rng.choice(int(n_cards * 0.8), int(rng.integers(1, 4)), replace=False)]
        ring_merchants = rng.choice(n_merchants, int(rng.integers(1, 3)), replace=False)
        start = rng.uniform(0, days - 1) * DAY
        for _ in range(int(rng.integers(15, 40))):
            rows.append(dict(timestamp=float(start + rng.uniform(0, 48 * 3600)),
                             card_id=f"card_{rng.choice(members)}", device_id=str(rng.choice(ring_devs)),
                             ip_id=str(rng.choice(ring_ips)), merchant_id=f"m_{rng.choice(ring_merchants)}",
                             amount=round(float(rng.lognormal(3.9, 0.8)), 2), label=1, scenario="ring"))

    for a in range(n_ato):
        card = int(rng.integers(0, n_cards))
        start = rng.uniform(1, days) * DAY
        for _ in range(int(rng.integers(1, 4))):
            rows.append(dict(timestamp=float(start + rng.uniform(0, 1800)), card_id=f"card_{card}",
                             device_id=f"dev_ato{a}", ip_id=f"ip_{rng.integers(0, int(n_cards * 0.8))}",
                             merchant_id=f"m_{rng.choice(n_merchants, p=merchant_pop)}",
                             amount=round(float(rng.lognormal(4.3, 0.9)), 2), label=1, scenario="ato"))

    rows.sort(key=lambda r: r["timestamp"])
    for i, row in enumerate(rows):
        row["txn_id"] = f"txn_{seed}_{i:07d}"
    return rows


async def stream(url: str, records: list[dict[str, Any]], rate: float, concurrency: int,
                 api_key: str | None) -> None:
    """POST records to the API at ``rate`` events/second and print latency stats."""
    import httpx  # imported lazily so training does not require it

    sem = asyncio.Semaphore(concurrency)
    client_ms: list[float] = []
    server_ms: list[float] = []
    alerts = errors = 0
    headers = {"X-API-Key": api_key} if api_key else {}

    async with httpx.AsyncClient(base_url=url, timeout=30.0, headers=headers) as client:
        async def send(rec: dict[str, Any]) -> None:
            nonlocal alerts, errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    resp = await client.post("/api/v1/transactions", json=rec)
                    resp.raise_for_status()
                    body = resp.json()
                    client_ms.append((time.perf_counter() - t0) * 1000)
                    server_ms.append(body["latency_ms"]["scoring_total"])
                    alerts += int(body["is_alert"])
                except Exception:  # noqa: BLE001 - count and continue streaming
                    errors += 1

        tasks = []
        interval = 1.0 / rate if rate > 0 else 0.0
        start = time.perf_counter()
        for rec in records:
            tasks.append(asyncio.create_task(send(rec)))
            if interval:
                await asyncio.sleep(interval)
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    def pct(values: list[float], q: float) -> float:
        return float(np.percentile(values, q)) if values else float("nan")

    print(f"sent={len(records)} ok={len(client_ms)} errors={errors} alerts={alerts} "
          f"throughput={len(client_ms) / elapsed:.1f} txn/s")
    if server_ms:
        print(f"server scoring ms  p50={pct(server_ms, 50):.2f} p95={pct(server_ms, 95):.2f} "
              f"p99={pct(server_ms, 99):.2f}")
        print(f"client round-trip ms p50={pct(client_ms, 50):.2f} p95={pct(client_ms, 95):.2f} "
              f"mean={statistics.mean(client_ms):.2f}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=2000, help="number of transactions to stream")
    p.add_argument("--rate", type=float, default=100.0, help="events per second (0 = as fast as possible)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=42, help="use a seed different from training")
    p.add_argument("--api-key", default=None)
    args = p.parse_args()
    records = generate_transactions(n_normal=args.n, n_rings=max(1, args.n // 1000),
                                    n_ato=max(2, args.n // 200), seed=args.seed)
    fraud = sum(r["label"] for r in records)
    print(f"streaming {len(records)} synthetic transactions ({fraud} fraud, {fraud / len(records):.2%})")
    asyncio.run(stream(args.url, records, args.rate, args.concurrency, args.api_key))


if __name__ == "__main__":
    main()
