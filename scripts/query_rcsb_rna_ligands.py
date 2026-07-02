import argparse
import json
import urllib.request
from pathlib import Path


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def terminal(attribute, operator, value):
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": attribute,
            "operator": operator,
            "value": value,
        },
    }


def build_query(args, start):
    nodes = [
        terminal("rcsb_entry_info.polymer_entity_count_RNA", "greater", 0),
        terminal("rcsb_entry_info.nonpolymer_entity_count", "greater", 0),
    ]
    if not args.allow_protein:
        nodes.append(terminal("rcsb_entry_info.polymer_entity_count_protein", "less_or_equal", 0))
    if args.max_polymer_monomers is not None:
        nodes.append(
            terminal(
                "rcsb_entry_info.polymer_monomer_count_maximum",
                "less_or_equal",
                args.max_polymer_monomers,
            )
        )
    if args.min_ligand_mw is not None:
        nodes.append(
            terminal(
                "rcsb_entry_info.nonpolymer_molecular_weight_maximum",
                "greater_or_equal",
                args.min_ligand_mw,
            )
        )
    if args.max_resolution is not None:
        nodes.append(
            terminal(
                "rcsb_entry_info.resolution_combined",
                "less_or_equal",
                args.max_resolution,
            )
        )
    if args.experimental_method:
        nodes.append(terminal("exptl.method", "exact_match", args.experimental_method))

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": start, "rows": args.page_size},
            "results_content_type": ["experimental"],
        },
    }


def post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def query_ids(args):
    ids = []
    total_count = None
    start = 0
    while len(ids) < args.max_results:
        payload = build_query(args, start)
        data = post_json(RCSB_SEARCH_URL, payload)
        if total_count is None:
            total_count = data.get("total_count", 0)
        result_set = data.get("result_set", [])
        if not result_set:
            break
        for item in result_set:
            identifier = item.get("identifier")
            if identifier:
                ids.append(identifier.lower())
                if len(ids) >= args.max_results:
                    break
        start += len(result_set)
        if start >= total_count:
            break
    return ids, total_count or 0


def main():
    parser = argparse.ArgumentParser(description="Query RCSB for RNA-ligand PDB IDs.")
    parser.add_argument("--out", default="data/pdb_contacts/pdb_ids.txt")
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--min-ligand-mw",
        type=float,
        default=0.15,
        help="Minimum maximum nonpolymer molecular weight in kDa. Use 0 to include ions/water-only entries.",
    )
    parser.add_argument(
        "--max-polymer-monomers",
        type=int,
        default=512,
        help="Maximum polymer monomer count. Use 0 to disable.",
    )
    parser.add_argument(
        "--allow-protein",
        action="store_true",
        help="Allow entries that also contain protein polymer chains.",
    )
    parser.add_argument(
        "--max-resolution",
        type=float,
        help="Optional resolution cutoff in Angstrom. Omit to keep NMR/no-resolution entries.",
    )
    parser.add_argument(
        "--experimental-method",
        help="Optional exact experimental method, e.g. 'X-RAY DIFFRACTION'.",
    )
    args = parser.parse_args()

    if args.max_results < 1:
        raise SystemExit("--max-results must be positive")
    if args.max_polymer_monomers == 0:
        args.max_polymer_monomers = None

    ids, total_count = query_ids(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"RCSB total_count={total_count} wrote={len(ids)} out={out_path}")


if __name__ == "__main__":
    main()
