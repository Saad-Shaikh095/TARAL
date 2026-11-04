from __future__ import annotations

import earthaccess
import requests


COLLECTION_ID = "C2098858642-POCLOUD"

GRANULE_ID = "G2183673179-POCLOUD"

CMR_URL = (
    "https://cmr.earthdata.nasa.gov/"
    "search/granules.json"
)


# ============================================================
# LOGIN
# ============================================================

print("=" * 70)
print("EARTHDATA LOGIN")
print("=" * 70)

auth = earthaccess.login(
    persist=True,
    strategy="interactive",
)

if not auth.authenticated:
    raise RuntimeError(
        "Earthdata authentication failed."
    )

print("Authentication successful.")


# ============================================================
# QUERY GRANULE
# ============================================================

print()
print("=" * 70)
print("QUERYING CMR")
print("=" * 70)

params = {
    "collection_concept_id": COLLECTION_ID,
    "granule_ur": "oscar_currents_final_20120101",
    "page_size": 10,
}

session = earthaccess.get_requests_https_session()

response = session.get(
    CMR_URL,
    params=params,
    timeout=60,
)

print(
    f"CMR HTTP status: {response.status_code}"
)

response.raise_for_status()

data = response.json()

entries = (
    data
    .get("feed", {})
    .get("entry", [])
)

print(
    f"Found {len(entries)} granule(s)."
)

if not entries:
    raise RuntimeError(
        "Could not find OSCAR granule."
    )


# ============================================================
# INSPECT
# ============================================================

granule = entries[0]

print()
print("=" * 70)
print("GRANULE INFORMATION")
print("=" * 70)

print(
    "ID:"
)

print(
    granule.get("id")
)

print()
print(
    "Producer granule:"
)

print(
    granule.get(
        "producer_granule_id"
    )
)

print()
print(
    "Temporal coverage:"
)

print(
    granule.get(
        "time_start"
    )
)

print(
    granule.get(
        "time_end"
    )
)

print()
print("=" * 70)
print("GRANULE LINKS")
print("=" * 70)

for link in granule.get(
    "links",
    [],
):

    href = link.get(
        "href"
    )

    rel = link.get(
        "rel"
    )

    title = link.get(
        "title"
    )

    hreflang = link.get(
        "hreflang"
    )

    print()
    print(
        f"Title    : {title}"
    )

    print(
        f"Relation : {rel}"
    )

    print(
        f"Language : {hreflang}"
    )

    print(
        f"URL      : {href}"
    )