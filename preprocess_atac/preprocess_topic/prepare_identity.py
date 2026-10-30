#!/usr/bin/env python3
"""Write identity sidecars from independently known mask and tile cell orders.

CSV: first column after header; NPY: one-dimensional string IDs; text: one ID
per line. The mask and tile files must contain matching cell sets.
"""
import argparse
import os
from pathlib import Path
import numpy as np

if __package__:
    from . import topic_inference as g
else:
    import topic_inference as g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mask-cell-ids', type=Path, required=True)
    p.add_argument('--tile-cell-ids', type=Path, required=True)
    p.add_argument('--normalization', choices=['exact', 'after_last_hash'], default='exact')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    g.require(not args.out.exists(), 'Identity output must be new')
    mask = g.read_ids(args.mask_cell_ids, args.normalization)
    tile = g.read_ids(args.tile_cell_ids, args.normalization)
    order = g.reorder_indices(tile, mask, 'tile versus mask identity')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Relative references make a complete input folder relocatable together.
    relative = lambda value: os.path.relpath(value.resolve(), args.out.parent.resolve())
    result = dict(mask_cell_ids=relative(args.mask_cell_ids),
                  tile_cell_ids=relative(args.tile_cell_ids), normalization=args.normalization,
                  n_mask_cells=len(mask), n_tile_cells=len(tile),
                  tile_reordering_needed=bool(np.any(order != np.arange(len(order)))))
    g.write_json(args.out, result)
    print(f'Identity sidecar saved: {args.out}; cells={len(mask)}')
    return result


if __name__ == '__main__':
    main()
