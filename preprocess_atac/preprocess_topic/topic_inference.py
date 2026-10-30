#!/usr/bin/env python3
"""Infer metacell topic distributions and aggregate genomic topic features.

Input cell IDs are aligned explicitly. MALLET tokens are resolved through the
trained alphabet; topic columns retain the trained dictionary order.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
from scipy import sparse

N_TOPICS = 50
MODES = ('equal', 'count')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(public_metadata(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(tmp, path)


def public_metadata(obj):
    """Write readable run metadata without internal content digests."""
    if isinstance(obj, dict):
        return {key: public_metadata(value) for key, value in obj.items()
                if 'sha256' not in str(key).lower() and not str(key).lower().endswith('_hash')}
    if isinstance(obj, (list, tuple)):
        return [public_metadata(value) for value in obj]
    return obj


def identity_hash(ids):
    h = hashlib.sha256()
    for value in ids:
        encoded = str(value).encode('utf-8')
        h.update(len(encoded).to_bytes(8, 'big'))
        h.update(encoded)
    return h.hexdigest()


def unique_ids(ids, label):
    ids = [str(x) for x in ids]
    require(all(ids) and len(set(ids)) == len(ids), label + ': empty or duplicate IDs')
    return ids


def reorder_indices(source_ids, target_ids, label):
    source_ids = unique_ids(source_ids, label + ' source')
    target_ids = unique_ids(target_ids, label + ' target')
    require(set(source_ids) == set(target_ids), label + ': missing/extra IDs')
    lookup = {value: i for i, value in enumerate(source_ids)}
    return np.asarray([lookup[value] for value in target_ids], dtype=np.int64)


def read_ids(path, normalization='exact'):
    path = Path(path)
    if path.suffix == '.npy':
        values = np.load(path, allow_pickle=False)
        require(values.ndim == 1, 'ID NPY must be one-dimensional')
        values = values.tolist()
    elif path.suffix == '.csv':
        with path.open(newline='') as f:
            reader = csv.reader(f)
            next(reader)  # CSV first column, after a mandatory header (e.g. LSI).
            values = [row[0] for row in reader if row]
    else:
        values = path.read_text().splitlines()
    values = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in values]
    if normalization == 'after_last_hash':
        values = [v.rsplit('#', 1)[-1] for v in values]
    else:
        require(normalization == 'exact', 'Unknown ID normalization')
    return unique_ids(values, str(path))


def load_mask(path, cell_ids, normalization='exact'):
    """CSV index is stable MC ID. Positional columns need an explicit ID sidecar."""
    rows, cols, values, mc_ids = [0], [], [], []
    with Path(path).open(newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        width = len(header) - 1
        require(width == len(cell_ids), 'Mask columns differ from supplied cell IDs')
        if header[1:] != [str(i) for i in range(width)]:
            labels = header[1:]
            if normalization == 'after_last_hash':
                labels = [v.rsplit('#', 1)[-1] for v in labels]
            require(labels == cell_ids, 'Named mask columns do not match ID sidecar order')
        for line, row in enumerate(reader, 2):
            require(len(row) == width + 1, f'Mask CSV malformed row {line}')
            mc_ids.append(row[0])
            for j, raw in enumerate(row[1:]):
                value = float(raw)
                require(value in (0., 1.), 'Mask must be finite binary membership')
                if value:
                    cols.append(j)
                    values.append(value)
            rows.append(len(cols))
    mc_ids = unique_ids(mc_ids, 'MC mask')
    require(len(mc_ids) > 0, 'No MCs')
    mask = sparse.csr_matrix((np.asarray(values, dtype=np.float64),
                             np.asarray(cols, dtype=np.int64),
                             np.asarray(rows, dtype=np.int64)), shape=(len(mc_ids), width))
    require(np.all(np.diff(mask.indptr) > 0), 'Mask contains empty MC membership')
    return mask, mc_ids


@dataclass
class Model:
    psi: np.ndarray
    region_names: list
    boundaries: dict
    topic_labels: list


def parse_regions(names):
    """Retain original row order; raw-global-token is its row index."""
    boundaries = {}
    current = None
    previous_end = None
    for i, name in enumerate(names):
        match = re.fullmatch(r'([^:]+):(\d+)-(\d+)', str(name))
        require(match is not None, f'Invalid region name at row {i}: {name}')
        chrom, start, end = match[1], int(match[2]), int(match[3])
        require(end - start == 500 and start % 500 == 0, 'Expected aligned 500bp regions')
        if chrom != current:
            require(chrom not in boundaries, 'Noncontiguous chromosome blocks in region_names')
            boundaries[chrom] = dict(start=i, end=i, n_bins=0, origin_bp=start)
            current = chrom
            previous_end = start
        require(start == previous_end, f'Gap/duplicate/unordered region in {chrom}: row {i}')
        previous_end = end
        boundaries[chrom]['end'] = i + 1
        boundaries[chrom]['n_bins'] += 1
    require(len(names) > 0, 'No model regions')
    return boundaries


def model_components(obj):
    if isinstance(obj, dict):
        names = obj.get('region_names')
        require(names is not None, 'region_names required; coordinate origin cannot be guessed')
        if obj.get('region_topic') is not None:
            psi = np.asarray(obj['region_topic'])
        else:
            require('topic_region' in obj, 'Model missing region_topic/topic_region')
            psi = np.asarray(obj['topic_region']).T
        if psi.shape != (len(names), N_TOPICS) and psi.shape == (N_TOPICS, len(names)):
            psi = psi.T
        require('topic_labels' in obj, 'Dictionary model needs explicit topic_labels; column order cannot be guessed')
        labels = list(obj['topic_labels'])
    else:
        require(hasattr(obj, 'topic_region'), 'Model missing topic_region')
        psi = obj.topic_region.values
        names = obj.topic_region.index.tolist()
        labels = obj.topic_region.columns.tolist()
    return psi, names, [str(x) for x in labels]


def validate_topic_order(labels):
    canonical = ([str(i) for i in range(N_TOPICS)],
                 [str(i) for i in range(1, N_TOPICS + 1)],
                 [f'Topic{i}' for i in range(1, N_TOPICS + 1)])
    require(list(map(str, labels)) in canonical,
            'Topic columns must be canonical MALLET order (0..49, 1..50, or Topic1..Topic50); '
            'permuted/unknown labels are rejected, never silently paired by column position')


def load_model(path):
    with Path(path).open('rb') as f:
        obj = pickle.load(f)
    psi, names, labels = model_components(obj)
    require(psi.shape == (len(names), N_TOPICS), 'Model must have 50 topics and matching region rows')
    require(len(labels) == N_TOPICS and len(set(map(str, labels))) == N_TOPICS,
            'Topic labels must be 50 unique values')
    validate_topic_order(labels)
    require(np.issubdtype(psi.dtype, np.number), 'Non-numeric psi')
    for start in range(0, len(names), 8192):
        block = psi[start:start + 8192]
        require(np.isfinite(block).all() and (block >= 0).all(), 'Invalid psi values')
    boundaries = parse_regions(names)
    require(all(b['origin_bp'] == 0 for b in boundaries.values()),
            'Nonzero chromosome origin is incompatible with existing absolute-bin dataloader; '
            'refusing cropped output (no automatic padding)')
    if isinstance(obj, dict) and 'chrom_boundaries' in obj:
        require(set(obj['chrom_boundaries']) == set(boundaries), 'Model chromosome metadata mismatch')
        for chrom, observed in boundaries.items():
            require(all(int(obj['chrom_boundaries'][chrom][k]) == observed[k]
                        for k in ('start', 'end', 'n_bins')), 'Model boundary/region-name mismatch')
    return Model(psi, names, boundaries, [str(x) for x in labels])


def nonnegative_sparse(matrix, label):
    require(sparse.issparse(matrix), label + ': sparse matrix required; refusing dense conversion')
    matrix = matrix.tocsr().astype(np.float64, copy=False)
    require(np.isfinite(matrix.data).all() and (matrix.data >= 0).all(), label + ': invalid counts')
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def build_profiles(tile_dict, tile_ids, mask, cell_ids, mc_ids, model, chromosomes, cache, batch_size):
    """Build sparse metacell batches and genome-wide coverage/depth scores."""
    cache.mkdir()
    n = len(mc_ids)
    coverage, depth = np.zeros(n), np.zeros(n)
    records = {chrom: [] for chrom in chromosomes}
    for chrom in chromosomes:
        require(chrom in tile_dict and chrom in tile_ids, 'Missing chromosome/row IDs: ' + chrom)
        raw = nonnegative_sparse(tile_dict.pop(chrom), chrom)
        require(raw.shape[0] == len(tile_ids[chrom]), chrom + ': tile ID/row count mismatch')
        order = reorder_indices(tile_ids[chrom], cell_ids, chrom + ' cell rows')
        # Reorder mask columns into the actual tile row order without a large tile copy.
        inverse = np.argsort(order)
        aligned_mask = mask[:, inverse]
        b = model.boundaries[chrom]
        lo = b['origin_bp'] // 500
        hi = lo + b['n_bins']
        require(raw.shape[1] >= hi, chrom + ': tile shorter than model coordinates')
        folder = cache / chrom
        folder.mkdir()
        for batch, start in enumerate(range(0, n, batch_size)):
            end = min(n, start + batch_size)
            counts = nonnegative_sparse(aligned_mask[start:end] @ raw, chrom + ' MC counts')
            coverage[start:end] += np.diff(counts.indptr)
            depth[start:end] += np.asarray(counts.sum(axis=1)).ravel()
            path = folder / f'batch_{batch:05d}.npz'
            sparse.save_npz(path, counts[:, lo:hi].tocsr())
            ids_path = path.with_suffix('.ids.json')
            write_json(ids_path, mc_ids[start:end])
            records[chrom].append(dict(path=str(path), ids_path=str(ids_path),
                                       start=start, end=end, batch=batch))
        del raw, aligned_mask
        print(f'PROFILE {chrom}: sparse MC shards ready', flush=True)
    require(np.isfinite(depth).all(), 'Global MC depth overflow')
    return records, coverage, depth


def load_profile(record, expected_ids):
    ids = record.get('mc_ids')
    if ids is None:
        ids = json.loads(Path(record['ids_path']).read_text())
    if 'matrix' in record:
        counts = record['matrix']
        require(all(not a.flags.writeable for a in (counts.data, counts.indices, counts.indptr)),
                'Shared memory cache must be read-only')
    else:
        if 'sha256' in record:
            require(sha256(record['path']) == record['sha256'], 'Profile cache hash mismatch')
        counts = nonnegative_sparse(sparse.load_npz(record['path']), 'cached MC counts')
    if 'csr_sha256' in record:
        require(csr_sha256(counts) == record['csr_sha256'], 'Profile content/identity mismatch')
    require(counts.shape[0] == len(ids), 'MC cache identity/row count mismatch')
    order = reorder_indices(ids, expected_ids, 'MC cache')
    return counts if np.array_equal(order, np.arange(len(ids))) else counts[order]


def csr_sha256(matrix):
    h = hashlib.sha256(str(matrix.shape).encode())
    for a in (matrix.indptr, matrix.indices, matrix.data):
        h.update(a.dtype.str.encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


def select_metacells(coverage, depth, mode, n_select, method):
    require(mode in ('all', 'topn'), 'Unknown selection mode')
    if mode == 'all':
        require(n_select is None, 'n_select_metacells is only valid with topn')
        return np.arange(len(coverage))
    n_select = n_select if n_select is not None else min(200, len(coverage))
    require(n_select > 0, 'n_select_metacells must be positive')
    if n_select >= len(coverage):
        return np.arange(len(coverage))
    c = (coverage - coverage.min()) / (np.ptp(coverage) + 1e-10)
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-10)
    require(method in ('coverage', 'depth', 'balance'), 'Unknown selection method')
    score = c if method == 'coverage' else d if method == 'depth' else c * d
    # Stable tie-breaking then canonical MC order, recorded explicitly.
    return np.sort(np.argsort(score, kind='stable')[-n_select:])


def write_global_documents(path, records, model, chromosomes, mc_ids, selected, batch, scratch):
    """Each document contains every requested chromosome for exactly one MC."""
    reference = records[chromosomes[0]][batch]
    start, end = reference['start'], reference['end']
    chosen = [int(i) for i in selected if start <= i < end]
    require(chosen, 'No selected MC in batch')
    scratch.mkdir()
    handles = {}
    try:
        for i in chosen:
            handles[i] = (scratch / str(i)).open('w')
            handles[i].write(f'metacell_{i} X ')
        for chrom in chromosomes:
            rec = records[chrom][batch]
            require((rec['start'], rec['end']) == (start, end), 'Inconsistent MC shard bounds')
            counts = load_profile(rec, mc_ids[start:end])
            require(counts.shape[1] == model.boundaries[chrom]['n_bins'], 'Cache/model bin mismatch')
            offset = model.boundaries[chrom]['start']
            for i in chosen:
                row = i - start
                indices = counts.indices[counts.indptr[row]:counts.indptr[row + 1]]
                for j in range(0, len(indices), 4096):
                    handles[i].write(' '.join(str(int(b) + offset) for b in indices[j:j + 4096]))
                    handles[i].write(' ')
        for handle in handles.values():
            handle.write('\n')
            handle.close()
        with path.open('wb') as output:
            for i in chosen:
                with (scratch / str(i)).open('rb') as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    finally:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(scratch)
    return [f'metacell_{i}' for i in chosen]


def parse_theta(path, expected_names):
    positions = {name: i for i, name in enumerate(expected_names)}
    result = np.zeros((len(expected_names), N_TOPICS), dtype=np.float64)
    seen = set()
    with Path(path).open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            require(len(parts) == N_TOPICS + 2, 'Expected dense 50-topic MALLET output')
            name = parts[1]
            require(name in positions and name not in seen, 'Unexpected/duplicate MALLET document: ' + name)
            values = np.asarray(parts[2:], dtype=np.float64)
            require(np.isfinite(values).all() and (values >= 0).all(), 'Invalid inferred theta')
            require(abs(values.sum() - 1.) < 1e-4, 'MALLET theta row does not sum to one')
            result[positions[name]] = values
            seen.add(name)
    require(seen == set(expected_names), 'Missing MALLET document rows')
    return result


class MalletBackend:
    def __init__(self, args):
        self.args = args
        executable = shutil.which(args.mallet_path) or str(Path(args.mallet_path).expanduser())
        home = Path(executable).resolve().parent.parent
        require((home / 'class').is_dir() and (home / 'lib/mallet-deps.jar').is_file(),
                'Expected MALLET bin/mallet, class/, lib/mallet-deps.jar layout')
        require(re.fullmatch(r'[1-9][0-9]*[mMgG]', args.mallet_memory) is not None, 'Invalid Java memory')
        java = shutil.which('java')
        require(java is not None, 'Java 11+ JDK required')
        self.java = [java, '-Xmx' + args.mallet_memory, '-ea', '-Djava.awt.headless=true',
                     '-Dfile.encoding=UTF-8', '--class-path',
                     str(home / 'class') + os.pathsep + str(home / 'lib/mallet-deps.jar')]
        self.inferencer_hash = sha256(args.inferencer_path)
        self.alphabet_hash = None

    def infer(self, corpus, names, folder, seed, max_region_index):
        helper = Path(__file__).parent / 'mallet/BuildAlignedMalletInput.java'
        aligned = folder / 'input.mallet'
        report = folder / 'alignment_report.json'
        cmd = self.java + [str(helper), '--inferencer', str(self.args.inferencer_path),
                '--input', str(corpus), '--output', str(aligned), '--report', str(report),
                '--oov-policy', 'error', '--expected-num-topics', str(N_TOPICS),
                '--max-region-index', str(max_region_index)]
        output = folder / 'doc_topics.tsv'
        infer_cmd = self.java + ['cc.mallet.topics.tui.InferTopics',
                '--inferencer', str(self.args.inferencer_path), '--input', str(aligned),
                '--output-doc-topics', str(output), '--num-iterations', str(self.args.mallet_iter),
                '--random-seed', str(seed), '--doc-topics-threshold', '0.0']
        write_json(folder / 'commands.json', dict(align=cmd, infer=infer_cmd, seed=seed))
        for name, command in [('align', cmd), ('infer', infer_cmd)]:
            with (folder / (name + '.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if name == 'align':
                audit = json.loads(report.read_text())
                require(audit['status'] == 'ok' and audit['inferencer_unchanged'], 'Alignment failed')
                require(sha256(self.args.inferencer_path) == self.inferencer_hash,
                        'Inferencer identity changed between batches')
                require(audit['totals']['documents'] == len(names) and
                        audit['totals']['oov_tokens'] == 0, 'Alignment document/OOV mismatch')
                # The immutable inferencer contains the exact trained alphabet.
                # Retain the alphabet mapping for in-memory
                # identity comparisons; public metadata never exposes it.
                alphabet_hash = self.inferencer_hash
                if self.alphabet_hash is None:
                    self.alphabet_hash = alphabet_hash
                require(self.alphabet_hash == alphabet_hash, 'Alphabet changed between batches')
        require(sha256(self.args.inferencer_path) == self.inferencer_hash, 'Inferencer changed during inference')
        return parse_theta(output, names)


def infer_global_theta(records, model, chromosomes, mc_ids, selected, out, seed, backend, keep_work=False):
    """One logical inference stage per cell: each selected whole-genome document once."""
    out.mkdir()
    theta = np.zeros((len(selected), N_TOPICS), dtype=np.float64)
    positions = {int(source): i for i, source in enumerate(selected)}
    batches = []
    assigned = set()
    for batch, record in enumerate(records[chromosomes[0]]):
        chosen = [int(i) for i in selected if record['start'] <= i < record['end']]
        if not chosen:
            continue
        folder = out / f'batch_{batch:05d}'
        folder.mkdir()
        corpus = folder / 'genomewide_docs.txt'
        names = write_global_documents(corpus, records, model, chromosomes, mc_ids,
                                       selected, batch, folder / 'spool')
        batch_seed = seed + batch
        require(batch_seed <= 2147483647, 'MALLET batch seed overflow')
        values = backend.infer(corpus, names, folder, batch_seed, len(model.region_names) - 1)
        require(values.shape == (len(chosen), N_TOPICS), 'Backend theta shape mismatch')
        require(np.isfinite(values).all() and (values >= 0).all() and
                np.all(np.abs(values.sum(axis=1) - 1.) < 1e-4), 'Backend theta invalid')
        for i, source in enumerate(chosen):
            require(source not in assigned, 'MC inferred more than once')
            theta[positions[source]] = values[i]
            assigned.add(source)
        batches.append(dict(batch=batch, seed=batch_seed, document_names=names,
                            mc_ids=[mc_ids[i] for i in chosen], corpus_sha256=sha256(corpus),
                            n_documents=len(chosen)))
        if not keep_work:
            corpus.unlink()
            (folder / 'input.mallet').unlink(missing_ok=True)
        print(f'THETA batch {batch}: {len(chosen)} whole-genome MC documents', flush=True)
    require(assigned == set(map(int, selected)), 'Not every selected MC inferred')
    write_json(out / 'batches.json', batches)
    return theta


def topk_contribution(theta_row, psi_rows, top_k):
    """Compute float32 products, normalize, select Top-k and renormalize."""
    require(1 <= top_k <= N_TOPICS, 'top_k must be in 1..50')
    normalized_theta = theta_row / (theta_row.sum() + 1e-10)
    product = (psi_rows * normalized_theta).astype(np.float32)
    require(np.isfinite(product).all(), 'theta*psi exceeds finite float32 range')
    row_sum = product.sum(axis=1, keepdims=True)
    q = np.divide(product, row_sum, out=np.zeros_like(product), where=row_sum > 0)
    indices = np.argsort(q, axis=1)[:, -top_k:]
    values = np.take_along_axis(q, indices, axis=1)
    denominator = values.sum(axis=1, keepdims=True)
    values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
    result = np.zeros_like(q)
    np.put_along_axis(result, indices, values, axis=1)
    return result, result.sum(axis=1) > 0


def new_mmap(path, shape):
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64, shape=shape)
    array[:] = 0.
    return array


def write_npz_members(path, members):
    """Members are .npy paths or tiny arrays; stream, never retain all chromosomes."""
    require(not path.exists(), 'Refusing existing output: ' + str(path))
    tmp = path.with_name(path.name + '.partial')
    with zipfile.ZipFile(tmp, 'x', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for key, value in members.items():
            with archive.open(key + '.npy', 'w', force_zip64=True) as target:
                if isinstance(value, Path):
                    with value.open('rb') as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                else:
                    np.lib.format.write_array(target, np.asarray(value), allow_pickle=False)
    os.replace(tmp, path)


def aggregate_chromosome(model, chrom, records, mc_ids, selected, theta, out, top_k, block_size):
    """Both numerators use each q once. Invalid q excludes MC from both denominators."""
    b = model.boundaries[chrom]
    n = b['n_bins']
    work = out / '_aggregate_work' / chrom
    work.mkdir(parents=True)
    nums = {mode: new_mmap(work / (mode + '_num.npy'), (n, N_TOPICS)) for mode in MODES}
    dens = {mode: new_mmap(work / (mode + '_den.npy'), (n,)) for mode in MODES}
    count_squared = new_mmap(work / 'count_squared.npy', (n,))
    positions = {int(source): i for i, source in enumerate(selected)}
    for rec in records[chrom]:
        chosen = [i for i in positions if rec['start'] <= i < rec['end']]
        if not chosen:
            continue
        counts = load_profile(rec, mc_ids[rec['start']:rec['end']])
        require(counts.shape[1] == n, 'Cached region shape mismatch')
        for source in chosen:
            i = source - rec['start']
            lo, hi = counts.indptr[i:i + 2]
            bins, weights = counts.indices[lo:hi], counts.data[lo:hi]
            for start in range(0, len(bins), block_size):
                indices = bins[start:start + block_size]
                w = weights[start:start + block_size]
                q, valid = topk_contribution(theta[positions[source]],
                    model.psi[b['start'] + indices], top_k)
                indices, q, w = indices[valid], q[valid], w[valid]
                nums['equal'][indices] += q
                dens['equal'][indices] += 1.
                nums['count'][indices] += q * w[:, None]
                dens['count'][indices] += w
                count_squared[indices] += w * w
    outputs, qc = {}, {}
    reliability_dir = out / 'reliability'
    reliability_dir.mkdir(exist_ok=True)
    # Independent sidecar: no new channels in model input NPZ.
    # Denominators are 1D per chromosome, never region x MC x topic.
    support = np.asarray(dens['equal']).astype(np.int64)
    weight_sum = np.asarray(dens['count']).copy()
    weight_sq_sum = np.asarray(count_squared).copy()
    np.savez_compressed(reliability_dir / (chrom + '_support.npz'),
        effective_mc_count=support, count_weight_sum=weight_sum,
        count_weight_squared_sum=weight_sq_sum,
        count_weight_neff=np.divide(weight_sum**2, weight_sq_sum,
            out=np.zeros_like(weight_sum), where=weight_sq_sum > 0),
        n_selected_mc=np.array(len(selected)), origin_bp=np.array(b['origin_bp']),
        bin_size_bp=np.array(500), n_bins=np.array(n))
    del support, weight_sum, weight_sq_sum
    for mode in MODES:
        directory = out / mode
        directory.mkdir(exist_ok=True)
        npy_path = directory / (chrom + '_matrix.npy')
        result = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.float16, shape=(n, N_TOPICS))
        zero_bins, max_row_sum_error = 0, 0.
        neff_sum = 0.
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            den = np.asarray(dens[mode][start:stop])
            num = np.asarray(nums[mode][start:stop])
            require(np.isfinite(num).all() and np.isfinite(den).all(), 'Aggregation overflow')
            values = np.divide(num, den[:, None], out=np.zeros_like(num), where=den[:, None] > 0)
            valid = den > 0
            if valid.any():
                max_row_sum_error = max(max_row_sum_error, float(np.max(np.abs(values[valid].sum(1) - 1))))
            zero_bins += int((~valid).sum())
            result[start:stop] = values.astype(np.float16)
            if mode == 'equal':
                neff_sum += float(den.sum())
            else:
                sq = np.asarray(count_squared[start:stop])
                require(np.isfinite(sq).all(), 'Squared count weights overflow')
                neff_sum += float(np.divide(den**2, sq, out=np.zeros_like(den), where=sq > 0).sum())
        require(max_row_sum_error < 1e-5, 'Aggregated valid row fails normalization')
        result.flush()
        del result
        write_npz_members(directory / (chrom + '_aggregated.npz'),
                          dict(matrix=npy_path, n_bins=np.array(n)))
        outputs[mode] = npy_path
        qc[mode] = dict(n_bins=n, zero_bins=zero_bins, mean_effective_contributors=neff_sum / n,
                        max_row_sum_error_before_float16=max_row_sum_error,
                        denominator='number_of_MC_with_C>0_and_nonzero_q' if mode == 'equal'
                        else 'sum_C_over_MC_with_C>0_and_nonzero_q')
    # Close mappings before removing temporary accumulator files (Windows-compatible).
    for a in list(nums.values()) + list(dens.values()) + [count_squared]:
        a.flush()
        a._mmap.close()
    shutil.rmtree(work)
    return outputs, qc


def build_outputs(model, chromosomes, records, mc_ids, selected, theta, out, cell_type, top_k, block_size):
    require(all(model.boundaries[c]['origin_bp'] == 0 for c in chromosomes),
            'Existing NPZ dataloader requires zero-origin chromosome matrices')
    if isinstance(theta, dict):
        require(set(theta) == set(chromosomes), 'Chromosome theta keys mismatch')
    for values in (theta.values() if isinstance(theta, dict) else [theta]):
        require(values.shape == (len(selected), N_TOPICS), 'Theta/selection mismatch')
        require(np.isfinite(values).all() and (values >= 0).all() and
                np.all(values.sum(axis=1) > 0), 'Invalid theta')
    members = {mode: {} for mode in MODES}
    qc = {}
    for chrom in chromosomes:
        outputs, qc[chrom] = aggregate_chromosome(model, chrom, records, mc_ids, selected,
                                                 theta[chrom] if isinstance(theta, dict) else theta,
                                                 out, top_k, block_size)
        for mode in MODES:
            members[mode][chrom + '_matrix'] = outputs[mode]
            members[mode][chrom + '_n_bins'] = np.array(model.boundaries[chrom]['n_bins'])
        print(f'AGGREGATE {chrom}: equal and count share theta/MC/TopK', flush=True)
    final = {}
    for mode in MODES:
        members[mode]['n_topics'] = np.array(N_TOPICS)
        members[mode]['chromosomes'] = np.asarray(chromosomes)
        path = out / mode / (cell_type + '_topics_aggregated.npz')
        write_npz_members(path, members[mode])
        final[mode] = dict(path=str(path), sha256=sha256(path))
    write_json(out / 'aggregation_qc.json', qc)
    return final


def load_identity_config(path, chromosomes):
    spec = json.loads(Path(path).read_text())
    norm = spec.get('normalization', 'exact')
    def resolve(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else Path(path).resolve().parent / p
    require('mask_cell_ids' in spec, 'identity JSON requires mask_cell_ids (e.g. LSI cell order)')
    paths = [resolve(spec['mask_cell_ids'])]
    mask_ids = read_ids(paths[0], norm)
    if 'mask_cell_ids_sha256' in spec:
        require(identity_hash(mask_ids) == spec['mask_cell_ids_sha256'], 'Mask ID sidecar changed')
    tile_ids = {}
    if 'tile_cell_ids_by_chromosome' in spec:
        mapping = spec['tile_cell_ids_by_chromosome']
        for chrom in chromosomes:
            require(chrom in mapping, 'Missing tile IDs for ' + chrom)
            p = resolve(mapping[chrom]); paths.append(p)
            tile_ids[chrom] = read_ids(p, norm)
    else:
        require('tile_cell_ids' in spec, 'identity JSON requires shared or per-chr tile IDs')
        p = resolve(spec['tile_cell_ids']); paths.append(p)
        ids = read_ids(p, norm)
        if 'tile_cell_ids_sha256' in spec:
            require(identity_hash(ids) == spec['tile_cell_ids_sha256'], 'Tile ID sidecar changed')
        tile_ids = {chrom: ids for chrom in chromosomes}
    return mask_ids, tile_ids, norm, list(dict.fromkeys(paths))


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'Input changed while hashing: ' + str(path))
    return dict(path=str(path), size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digest)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell-type', '--cell_type', required=True)
    p.add_argument('--tile-file', '--tile_file', type=Path)
    p.add_argument('--mask-file', '--mask_file', type=Path)
    p.add_argument('--tile-data-dir', '--tile_data_dir', type=Path)
    p.add_argument('--model-dir', '--model_dir', type=Path)
    p.add_argument('--identity-json', '--identity_json', type=Path, required=True)
    p.add_argument('--model-file', '--model_file', type=Path)
    p.add_argument('--inferencer-path', '--inferencer_path', type=Path)
    p.add_argument('--chromosomes', nargs='+', required=True)
    p.add_argument('--genome', default='hg38', choices=['hg38','hg19','mm10'])
    p.add_argument('--output-dir', '--output_dir', type=Path, required=True)
    p.add_argument('--mallet-path', '--mallet_path', default='mallet')
    p.add_argument('--mallet-memory', '--mallet_memory', default='16G')
    p.add_argument('--mallet-iter', '--mallet_iter', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--mc-batch-size', '--mc_batch_size', type=int, default=32)
    p.add_argument('--region-block-size', '--region_block_size', type=int, default=4096)
    p.add_argument('--top-k', '--top_k', type=int, default=5)
    p.add_argument('--aggregation-mode', '--aggregation_mode', choices=['all','topn'], default='all',
                   help='MC selection only; both equal and count outputs are always produced')
    p.add_argument('--n-select-metacells', '--n_select_metacells', type=int)
    p.add_argument('--selection-method', '--selection_method', choices=['coverage','depth','balance'], default='balance')
    p.add_argument('--keep-work', action='store_true', help='Keep large text/MALLET/cached-count/NPY work artifacts')
    args = p.parse_args(argv)
    if args.tile_data_dir is not None:
        require(args.tile_file is None and args.mask_file is None,
                'Choose --tile-data-dir or explicit --tile-file/--mask-file')
        args.tile_file = args.tile_data_dir / f'{args.cell_type}_tile_500bp_dict.p'
        args.mask_file = args.tile_data_dir / f'{args.cell_type}_metacell_mask.csv'
    if args.model_dir is not None:
        require(args.model_file is None, 'Choose --model-dir or --model-file')
        args.model_file = args.model_dir / 'Topic50.pkl'
        if args.inferencer_path is None:
            args.inferencer_path = args.model_dir / 'inferencer.mallet'
    for field in ('tile_file', 'mask_file', 'model_file', 'inferencer_path'):
        require(getattr(args, field) is not None, 'Missing --' + field.replace('_', '-'))
    del args.tile_data_dir, args.model_dir
    require(re.fullmatch(r'[A-Za-z0-9_-]+', args.cell_type) is not None, 'Invalid cell type filename')
    require(len(set(args.chromosomes)) == len(args.chromosomes), 'Duplicate chromosomes')
    require(all(re.fullmatch(r'chr[0-9XYM]+', c) for c in args.chromosomes), 'Invalid chromosome name')
    require(1 <= args.top_k <= N_TOPICS, 'top_k must be 1..50')
    require(args.mc_batch_size > 0 and args.region_block_size > 0 and args.mallet_iter > 0,
            'Batch/block/iteration sizes must be positive')
    require(1 <= args.seed <= 2147483647, 'Seed must be a positive Java int')
    require(args.aggregation_mode == 'topn' or args.n_select_metacells is None,
            'n-select-metacells requires topn')
    if args.n_select_metacells is not None:
        require(args.n_select_metacells > 0, 'n-select-metacells must be positive')
    return args


def run(args, backend=None):
    out = args.output_dir.resolve()
    require(not out.exists(), 'Output directory must be new; partial/old runs are never reused')
    out.mkdir(parents=True)
    manifest = dict(status='running', schema_version=1, cell_type=args.cell_type,
        theta_scope='genomewide_shared_per_metacell', theta_chromosomes=args.chromosomes,
        genome=args.genome, n_topics=N_TOPICS,
        aggregate_modes=list(MODES), count_weight_definition='C_bin_MC_after_mask_times_tile',
        count_weight_caveat='Unnormalized counts may reflect MC size and sequencing depth; no exposure correction.',
        atac_protocol='ATAC input from all listed chromosomes; no Hi-C labels read; dictionary training scope unchanged.',
        seed=args.seed, batch_seed_rule='seed + original MC shard index',
        mc_batch_size=args.mc_batch_size, top_k=args.top_k,
        selection=dict(mode=args.aggregation_mode, n_select=args.n_select_metacells, method=args.selection_method,
                       score_scope='full tile span of requested chromosomes; stable ties, canonical MC order'),
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        started_unix=time.time(), source_sha256=sha256(__file__),
        java_helper_sha256=sha256(Path(__file__).parent/'mallet/BuildAlignedMalletInput.java'),
        software=dict(python=sys.version, numpy=np.__version__, scipy=__import__('scipy').__version__))
    write_json(out / 'manifest.json', manifest)
    try:
        cell_ids, tile_ids, norm, id_paths = load_identity_config(args.identity_json, args.chromosomes)
        input_paths = [args.tile_file,args.mask_file,args.identity_json,args.model_file,args.inferencer_path] + id_paths
        manifest['inputs'] = [fingerprint(p) for p in dict.fromkeys(input_paths)]
        input_map = {r['path']:r['sha256'] for r in manifest['inputs']}
        manifest['dictionary_identity'] = dict(model_sha256=input_map[str(args.model_file.resolve())],
                            inferencer_sha256=input_map[str(args.inferencer_path.resolve())],
                            model_pair='caller-supplied model and trained inferencer')
        model = load_model(args.model_file)
        require(all(c in model.boundaries for c in args.chromosomes), 'Requested chr absent from model')
        manifest['model_boundaries'] = model.boundaries
        manifest['topic_labels'] = model.topic_labels
        manifest['region_names_order_sha256'] = identity_hash(map(str, model.region_names))
        manifest['requested_scope_covers_all_model_chromosomes'] = set(args.chromosomes) == set(model.boundaries)
        mask, mc_ids = load_mask(args.mask_file, cell_ids, norm)
        manifest['mc_ids_sha256'] = identity_hash(mc_ids)
        manifest['mask_cell_ids_sha256'] = identity_hash(cell_ids)
        manifest['tile_cell_ids_sha256_by_chr'] = {c:identity_hash(tile_ids[c]) for c in args.chromosomes}
        write_json(out/'mc_ids.json', mc_ids)
        write_json(out/'manifest.json', manifest)
        with args.tile_file.open('rb') as f:
            tiles = pickle.load(f)
        require(isinstance(tiles,dict), 'Expected chromosome-keyed tile dictionary')
        records, coverage, depth = build_profiles(tiles, tile_ids, mask, cell_ids, mc_ids,
                    model, args.chromosomes, out/'profiles', args.mc_batch_size)
        del tiles
        selected = select_metacells(coverage, depth, args.aggregation_mode,
                                    args.n_select_metacells, args.selection_method)
        require(len(selected) > 0, 'Empty MC selection')
        np.savez_compressed(out/'mc_statistics.npz', mc_ids=np.asarray(mc_ids),
                            size=np.diff(mask.indptr), coverage=coverage, depth=depth,
                            selected_indices=selected, selected_mc_ids=np.asarray(mc_ids)[selected])
        del mask
        manifest['selected_mc_ids'] = [mc_ids[int(i)] for i in selected]
        manifest['selected_mc_ids_sha256'] = identity_hash(manifest['selected_mc_ids'])
        write_json(out/'manifest.json', manifest)
        backend = backend if backend is not None else MalletBackend(args)
        theta = infer_global_theta(records, model, args.chromosomes, mc_ids, selected,
                                  out/'theta_batches', args.seed, backend, args.keep_work)
        theta_path = out/'shared_theta.npz'
        np.savez_compressed(theta_path, theta=theta, mc_ids=np.asarray(mc_ids)[selected],
                            mask_row_indices=selected, chromosomes=np.asarray(args.chromosomes),
                            theta_scope=np.array('genomewide_shared_per_metacell'))
        manifest['shared_theta_sha256'] = sha256(theta_path)
        manifest['alphabet_sha256'] = getattr(backend,'alphabet_hash',None)
        manifest['theta_inference_stage_count'] = 1
        manifest['n_unique_inferred_MC_documents'] = len(selected)
        write_json(out/'manifest.json', manifest)
        manifest['outputs'] = build_outputs(model,args.chromosomes,records,mc_ids,selected,theta,out,
                                            args.cell_type,args.top_k,args.region_block_size)
        for record in manifest['inputs']:
            stat = Path(record['path']).stat()
            require((stat.st_size,stat.st_mtime_ns) == (record['size'],record['mtime_ns']),
                    'Input metadata changed during run: ' + record['path'])
        for path in (args.model_file,args.inferencer_path):
            require(sha256(path) == input_map[str(path.resolve())], 'Dictionary artifact changed during run')
        if not args.keep_work:
            shutil.rmtree(out/'profiles')
            for mode in MODES:
                for chrom in args.chromosomes:
                    (out/mode/(chrom+'_matrix.npy')).unlink()
        manifest.update(status='complete', finished_unix=time.time())
        write_json(out/'manifest.json', manifest)
        (out/'SUCCESS').write_text('Shared genomewide theta and both aggregates complete\n')
        return public_metadata(manifest)
    except BaseException as error:
        manifest.update(status='failed',error=type(error).__name__ + ': ' + str(error),finished_unix=time.time())
        write_json(out/'manifest.json',manifest)
        raise


def main(argv=None):
    return run(parse_args(argv))


if __name__ == '__main__':
    main()
