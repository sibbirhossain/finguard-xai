"""Loader for the public IEEE-CIS Fraud Detection dataset (Kaggle, Vesta Corporation).

The raw files are NOT redistributed; accept the competition rules on Kaggle and
download ``train_transaction.csv`` and ``train_identity.csv`` yourself.

Entity mapping onto FinGuard-XAI's four node types (documented assumption):
  card     -> card1|card2|card3|card5|addr1   (common "card-holder" proxy key)
  device   -> DeviceInfo (+ DeviceType); "unknown" when missing
  merchant -> ProductCD|addr2                  (product/region proxy; no merchant id is published)
  ip       -> P_emaildomain                    (network-identity proxy; no IP is published)
Missing identifiers are made unique per transaction so they never create false links.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

TXN_COLS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
            "card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain"]
ID_COLS = ["TransactionID", "DeviceInfo", "DeviceType"]


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(object).fillna("NA").astype(str).agg("|".join, axis=1)


def load_ieee_cis(data_dir: Union[str, Path], max_rows: int | None = None) -> list[dict[str, Any]]:
    """Return chronologically ordered records in the FinGuard-XAI schema."""
    data_dir = Path(data_dir)
    txn = pd.read_csv(data_dir / "train_transaction.csv", usecols=TXN_COLS, nrows=max_rows)
    ident_path = data_dir / "train_identity.csv"
    if ident_path.exists():
        txn = txn.merge(pd.read_csv(ident_path, usecols=ID_COLS), on="TransactionID", how="left")
    else:
        txn["DeviceInfo"], txn["DeviceType"] = np.nan, np.nan
    txn = txn.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    tid = txn["TransactionID"].astype(str)

    card = "card_" + _key(txn, ["card1", "card2", "card3", "card5", "addr1"])
    has_dev = txn["DeviceInfo"].notna()
    device = np.where(has_dev, "dev_" + _key(txn, ["DeviceInfo", "DeviceType"]),
                      "dev_missing_" + tid)
    merchant = "m_" + _key(txn, ["ProductCD", "addr2"])
    has_email = txn["P_emaildomain"].notna()
    ip = np.where(has_email, "email_" + txn["P_emaildomain"].astype(object).fillna("NA").astype(str), "email_missing_" + tid)

    return [dict(txn_id=f"ieee_{t}", timestamp=float(ts), card_id=c, device_id=d, merchant_id=m, ip_id=i,
                 amount=float(a), label=int(y), scenario="fraud" if y else "normal")
            for t, ts, c, d, m, i, a, y in zip(tid, txn["TransactionDT"], card, device, merchant, ip,
                                               txn["TransactionAmt"], txn["isFraud"])]
