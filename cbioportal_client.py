"""cBioPortal client for the MTAP-loss candidate reviewer — 0th step.

Connects to the public cBioPortal API, auto-discovers a study's molecular
profiles (never hardcoded), pulls MTAP/CDKN2A/CDKN2B GISTIC CNA + MTAP RNA-seq,
merges them per sample, caches every pull to CSV, and prints the viability
counts that decide whether the rest of the plan is worth building.

Run:  python cbioportal_client.py
"""

from __future__ import annotations

import hashlib
import os
import time

import pandas as pd
import requests

API_URL = "https://www.cbioportal.org/api"
# bravado only supports Swagger 2.0; the v3 endpoint is OpenAPI 3 and silently
# returns None from calls. The v2 endpoint serves a Swagger 2.0 spec. This is
# the LEGACY, non-canonical endpoint — cBioPortal's documented/maintained API
# is v3, so this could be deprecated without notice; the requests-based
# fallback below hits v3 directly and isn't affected by that risk.
SWAGGER_URL = f"{API_URL}/v2/api-docs"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

MTAP, CDKN2A, CDKN2B = 4507, 1029, 1030
GENE_NAMES = {MTAP: "MTAP", CDKN2A: "CDKN2A", CDKN2B: "CDKN2B"}


# --------------------------------------------------------------------------- #
# Cache validity (content-hash gated) + retry helpers
# --------------------------------------------------------------------------- #
def _sig(*parts):
    """Short content hash of the request parameters behind a cache file."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _cache_valid(cache_path, sig):
    """True if cache_path exists AND its recorded signature matches `sig`.

    A cache with no sidecar signature (pre-existing caches from before this
    check existed) is grandfathered in as valid rather than force-invalidated
    — but the signature is backfilled so future parameter changes ARE caught.
    """
    if not os.path.exists(cache_path):
        return False
    sig_path = cache_path + ".sig"
    if not os.path.exists(sig_path):
        with open(sig_path, "w") as f:
            f.write(sig)
        return True
    return open(sig_path).read().strip() == sig


def _write_sig(cache_path, sig):
    with open(cache_path + ".sig", "w") as f:
        f.write(sig)


def _retry(fn, attempts=3, base_delay=1.0, what="request"):
    """Retry `fn()` with exponential backoff on any exception. No retry logic
    existed before — a single transient network blip killed the whole pull."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network calls fail many ways
            last_exc = exc
            if i < attempts - 1:
                delay = base_delay * (2 ** i)
                print(f"[retry] {what} failed ({exc!r}); retrying in {delay:.1f}s "
                      f"({i + 1}/{attempts})")
                time.sleep(delay)
    raise last_exc


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def connect():
    """Return a bravado SwaggerClient, or None to signal the requests fallback."""
    try:
        from bravado.client import SwaggerClient

        client = SwaggerClient.from_url(
            SWAGGER_URL,
            config={
                "validate_requests": False,
                "validate_responses": False,
                "validate_swagger_spec": False,
            },
        )
        # Smoke test: bravado can build a client from an incompatible spec yet
        # return None from every call. Verify a real call works before trusting it.
        probe = client.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(
            studyId="blca_tcga_pan_can_atlas_2018"
        ).result()
        if not probe:
            raise RuntimeError("bravado built but returned no data (spec mismatch)")
        print(f"[connect] bravado client built and verified from {SWAGGER_URL}")
        return client
    except Exception as exc:  # noqa: BLE001 - bravado/swagger can fail many ways
        print(f"[connect] bravado unavailable ({exc!r}); using requests fallback")
        return None


# --------------------------------------------------------------------------- #
# Profile picker (never hardcode profile ids)
# --------------------------------------------------------------------------- #
def _list_profiles(client, study):
    """Return list of (molecularProfileId, name) for a study."""
    if client is not None:
        profiles = client.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(
            studyId=study
        ).result()
        return [(p.molecularProfileId, p.name) for p in profiles]
    resp = requests.get(f"{API_URL}/studies/{study}/molecular-profiles", timeout=60)
    resp.raise_for_status()
    return [(p["molecularProfileId"], p.get("name", "")) for p in resp.json()]


def pick_profile(client, study, substring):
    """Pick the molecular profile id whose id contains `substring`.

    Raises with the full available list if nothing matches.
    """
    profiles = _list_profiles(client, study)
    matches = [pid for pid, _ in profiles if substring in pid]
    if not matches:
        available = "\n  ".join(f"{pid}  ({name})" for pid, name in profiles)
        raise ValueError(
            f"No molecular profile in '{study}' contains '{substring}'.\n"
            f"Available profiles:\n  {available}"
        )
    if len(matches) > 1:
        print(f"[pick_profile] '{substring}' matched {matches}; using {matches[0]}")
    return matches[0]


# --------------------------------------------------------------------------- #
# Hugo symbol -> Entrez resolution (cache-first)
# --------------------------------------------------------------------------- #
def resolve_symbols_to_entrez(client, symbols, cache_name="gene_map"):
    """Resolve Hugo symbols to Entrez IDs; cache to CSV.

    Returns a DataFrame with columns: hugoGeneSymbol, entrezGeneId.
    Symbols the API cannot resolve are simply absent (logged by the caller).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.csv")
    sig = _sig("genes", sorted(symbols))
    if _cache_valid(cache_path, sig):
        print(f"[genes] cache hit: {cache_path}")
        return pd.read_csv(cache_path)

    if client is not None:
        recs = _retry(
            lambda: client.Genes.fetchGenesUsingPOST(
                geneIdType="HUGO_GENE_SYMBOL", geneIds=list(symbols)
            ).result(),
            what="gene resolution (bravado)",
        )
        rows = [
            {"hugoGeneSymbol": g.hugoGeneSymbol, "entrezGeneId": g.entrezGeneId}
            for g in recs
        ]
    else:
        def _do_fetch():
            resp = requests.post(
                f"{API_URL}/genes/fetch",
                params={"geneIdType": "HUGO_GENE_SYMBOL"},
                json=list(symbols),
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        rows = [
            {"hugoGeneSymbol": g["hugoGeneSymbol"], "entrezGeneId": g["entrezGeneId"]}
            for g in _retry(_do_fetch, what="gene resolution (requests)")
        ]

    df = pd.DataFrame(rows, columns=["hugoGeneSymbol", "entrezGeneId"])
    df.to_csv(cache_path, index=False)
    _write_sig(cache_path, sig)
    print(f"[genes] resolved {len(df)}/{len(symbols)} symbols -> cached {cache_path}")
    return df


# --------------------------------------------------------------------------- #
# Fetch (cache-first)
# --------------------------------------------------------------------------- #
def fetch_molecular_data(client, study, profile_id, entrez_ids, cache_name):
    """Fetch molecular data for genes in a profile; cache to CSV.

    Returns a DataFrame with columns: sampleId, entrezGeneId, value.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.csv")
    sig = _sig(study, profile_id, sorted(entrez_ids))
    if _cache_valid(cache_path, sig):
        print(f"[fetch] cache hit: {cache_path}")
        return pd.read_csv(cache_path)

    data_filter = {
        "entrezGeneIds": list(entrez_ids),
        "sampleListId": f"{study}_all",
    }

    if client is not None:
        records = _retry(
            lambda: client.Molecular_Data.fetchAllMolecularDataInMolecularProfileUsingPOST(
                molecularProfileId=profile_id,
                molecularDataFilter=data_filter,
                projection="SUMMARY",
            ).result(),
            what=f"molecular data fetch (bravado, {cache_name})",
        )
        rows = [
            {"sampleId": r.sampleId, "entrezGeneId": r.entrezGeneId, "value": r.value}
            for r in records
        ]
    else:
        def _do_fetch():
            url = f"{API_URL}/molecular-profiles/{profile_id}/molecular-data/fetch"
            resp = requests.post(
                url,
                params={"projection": "SUMMARY"},
                json=data_filter,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        rows = [
            {"sampleId": r["sampleId"], "entrezGeneId": r["entrezGeneId"], "value": r["value"]}
            for r in _retry(_do_fetch, what=f"molecular data fetch (requests, {cache_name})")
        ]

    df = pd.DataFrame(rows, columns=["sampleId", "entrezGeneId", "value"])
    df.to_csv(cache_path, index=False)
    _write_sig(cache_path, sig)
    print(f"[fetch] pulled {len(df)} records -> cached {cache_path}")
    return df


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def merge_cna_expression(study, cna_df, rna_df):
    """Pivot CNA (3 genes) to per-sample columns and attach MTAP RNA-seq."""
    cna = cna_df.copy()
    cna["gene"] = cna["entrezGeneId"].map(GENE_NAMES)
    wide = cna.pivot_table(
        index="sampleId", columns="gene", values="value", aggfunc="first"
    )
    wide = wide.rename(columns={g: f"{g}_cna" for g in ["MTAP", "CDKN2A", "CDKN2B"]})

    rna = rna_df[rna_df["entrezGeneId"] == MTAP][["sampleId", "value"]]
    rna = rna.rename(columns={"value": "MTAP_rna"}).set_index("sampleId")

    merged = wide.join(rna, how="left").reset_index()

    os.makedirs(CACHE_DIR, exist_ok=True)
    merged_path = os.path.join(CACHE_DIR, f"merged_{study}.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[merge] wrote {merged_path} ({len(merged)} samples)")
    return merged


# --------------------------------------------------------------------------- #
# Counts (the whole point of the 0th step)
# --------------------------------------------------------------------------- #
def _bucket(v):
    if pd.isna(v):
        return None
    v = int(v)
    if v == -2:
        return "-2 (homozygous deletion / positive)"
    if v == -1:
        return "-1 (shallow loss / ambiguous)"
    return "{0,1,2} (non-deleted reference)"


def print_counts(merged, study, cna_profile, rna_profile):
    m = merged.copy()
    m = m[m["MTAP_cna"].notna()]  # samples with MTAP GISTIC data
    m["bucket"] = m["MTAP_cna"].map(_bucket)
    has_rna = m["MTAP_rna"].notna()

    print("\n" + "=" * 68)
    print(f"VIABILITY COUNTS  —  {study}")
    print("=" * 68)
    print(f"CNA profile (auto-picked): {cna_profile}")
    print(f"RNA profile (auto-picked): {rna_profile}")
    print(f"\n# samples with MTAP GISTIC data: {len(m)}")
    print(f"# samples with matched MTAP RNA-seq: {int(has_rna.sum())}")

    order = [
        "-2 (homozygous deletion / positive)",
        "-1 (shallow loss / ambiguous)",
        "{0,1,2} (non-deleted reference)",
    ]
    print(f"\n{'MTAP CNA group':<42}{'n':>7}{'with RNA-seq':>15}")
    print("-" * 64)
    for b in order:
        grp = m[m["bucket"] == b]
        print(f"{b:<42}{len(grp):>7}{int(grp['MTAP_rna'].notna().sum()):>15}")
    print("=" * 68)

    pos = len(m[m["bucket"] == order[0]])
    amb_rna = int(m[m["bucket"] == order[1]]["MTAP_rna"].notna().sum())
    print("\nDECISION GATE:")
    print(f"  positives (-2): {pos}  (want >= ~15)")
    print(f"  ambiguous (-1) with expression: {amb_rna}  (want >= ~20)")
    ok = pos >= 15 and amb_rna >= 20
    print(f"  => {'PROCEED with this cohort.' if ok else 'TOO SMALL — consider switching/combining cohorts.'}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    study = "blca_tcga_pan_can_atlas_2018"
    client = connect()

    cna_profile = pick_profile(client, study, "gistic")
    rna_profile = pick_profile(client, study, "rna_seq_v2_mrna")
    print(f"[main] CNA profile:  {cna_profile}")
    print(f"[main] RNA profile:  {rna_profile}")

    cna_df = fetch_molecular_data(
        client, study, cna_profile, [MTAP, CDKN2A, CDKN2B], f"cna_{study}"
    )
    rna_df = fetch_molecular_data(
        client, study, rna_profile, [MTAP], f"rna_{study}"
    )

    merged = merge_cna_expression(study, cna_df, rna_df)
    print_counts(merged, study, cna_profile, rna_profile)


if __name__ == "__main__":
    main()
