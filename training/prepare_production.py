"""Build conservative weak labels from frozen, reviewed production records. Stdlib only."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path


def compact(value):
    return ''.join(str(value or '').split())


def ascii_text(value):
    return ''.join(' ' if c.isspace() else c for c in str(value or '') if c.isspace() or ' ' <= c <= '~')


def subsequence(raw, clean):
    it = iter(raw)
    return all(any(c == x for x in it) for c in clean)


def align(target, blocks, previous=None):
    """Unique exact match first; an evidenced previous span can disambiguate repeats."""
    target = compact(target)
    if not target:
        return []
    candidates = [b['ocr_index'] for b in blocks if target in compact(b['text'])]
    if len(candidates) == 1:
        return candidates
    previous = previous or {}
    ids = previous.get('ocr_indices') or []
    by_id = {b['ocr_index']: b for b in blocks}
    if ids and len(ids) == len(set(ids)) and all(i in by_id for i in ids):
        raw = ''.join(by_id[i]['text'] for i in ids)
        if target in compact(raw):
            return ids
    return []


def split_groups(records, excluded_hashes):
    # Group repeated scans by image SHA and reviewed tracking number before time splitting.
    parent = list(range(len(records)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, d in enumerate(records):
        sha = d.get('request', {}).get('sha256')
        track = compact(d.get('review', {}).get('data', {}).get('tracking_no'))
        keys = ([('sha', sha)] if sha else []) + ([('track', track)] if len(track) >= 8 else [])
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[root(i)].append(i)
    held = {g for g, ids in groups.items() if any(records[i].get('request', {}).get('sha256') in excluded_hashes for i in ids)}
    def date(i):
        value = records[i].get('created_at', '')
        return value.get('$date', '') if isinstance(value, dict) else str(value)
    ordered = sorted((g for g in groups if g not in held), key=lambda g: (max(date(i) for i in groups[g]), g))
    splits = {}
    for rank, g in enumerate(ordered):
        split = 'train' if rank < int(.7 * len(ordered)) else 'validation' if rank < int(.85 * len(ordered)) else 'test'
        for i in groups[g]:
            splits[records[i]['_id']] = (split, str(g))
    for g in held:
        for i in groups[g]:
            splits[records[i]['_id']] = ('llm_holdout', str(g))
    return splits


def prepare(record):
    data = record['review']['data']
    blocks = record.get('predict', {}).get('items') or []
    by_id = {b['ocr_index']: b for b in blocks}
    if len(by_id) != len(blocks):
        raise ValueError('Duplicate OCR indices')
    old = record.get('postprocess') or {}
    entities, cleaners, issues = [], [], []
    def add(kind, target, previous=None):
        if not isinstance(target, str) or not compact(target):
            issues.append(kind + ':missing_target')
            return []
        ids = align(target, blocks, previous)
        if not ids:
            issues.append(kind + ':unaligned')
            return []
        entities.append((kind, ids))
        raw = ''.join(by_id[i]['text'] for i in ids)
        clean = compact(target) if kind in ('TRACK', 'SKU') else target.strip()
        raw, clean = ascii_text(raw), ascii_text(clean)
        if clean and subsequence(raw, clean):
            cleaners.append({'kind': kind.lower(), 'raw': raw, 'clean': clean, 'ocr_indices': json.dumps(ids)})
        return ids
    add('TRACK', data.get('tracking_no'), old.get('track'))
    add('RECIPIENT', data.get('recipient'), old.get('recipient'))
    items = data.get('items')
    if not isinstance(items, list):
        issues.append('items:missing')
        items = []
    for item in items:
        sku = item.get('sku')
        candidates = [p for p in old.get('items', []) if compact(p.get('sku', {}).get('clean')) == compact(sku)]
        previous = candidates[0].get('sku') if len(candidates) == 1 else None
        ids = add('SKU', sku, previous)
        quantity = str(item.get('qty', ''))
        if not ids or not quantity.isascii() or not quantity.isdecimal():
            issues.append('QTY:missing_or_unaligned_sku')
            continue
        rects = [by_id[i]['bbox']['rect_px'] for i in ids]
        y0, y1 = min(r[1] for r in rects), max(r[3] for r in rects)
        qty_candidates = []
        for b in blocks:
            rect = b['bbox']['rect_px']
            overlap = min(y1, rect[3]) - max(y0, rect[1])
            if b['ocr_index'] not in ids and compact(b['text']) == quantity and overlap > .5 * max(1, min(y1-y0, rect[3]-rect[1])):
                qty_candidates.append(b['ocr_index'])
        if len(qty_candidates) == 1:
            entities.append(('QTY', qty_candidates))
        else:
            issues.append('QTY:ambiguous_or_missing')
    labels = {}
    for kind, ids in entities:
        for n, i in enumerate(ids):
            if i in labels:
                issues.append('overlapping_entities')
            labels[i] = ('B-' if n == 0 else 'I-') + kind
    # Only fully aligned documents get O labels. Missing fields are never silently negatives.
    if not issues and blocks:
        labels = {b['ocr_index']: labels.get(b['ocr_index'], 'O') for b in blocks}
    return labels, cleaners, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.records, 'rt') as f:
        metadata = json.loads(next(f))
        all_records = [json.loads(line) for line in f]
    records = [d for d in all_records if d.get('status') == 'reviewed' and isinstance((d.get('review') or {}).get('data'), dict)]
    excluded = {s['sha256'] for p in args.exclude_manifest for s in json.loads(p.read_text())['samples']}
    splits = split_groups(records, excluded)
    stats, reasons, clean_rows, docs = Counter(), Counter(), [], []
    clean_seen = set()
    for record in records:
        split, group = splits[record['_id']]
        stats['records_' + split] += 1
        labels, rows, issues = prepare(record)
        reasons.update(set(issues))
        for row in rows:
            # Deduplicate identical supervision within each split; preserve cross-split counts for audit.
            key = (split, row['kind'], row['raw'], row['clean'])
            if key in clean_seen:
                continue
            clean_seen.add(key)
            clean_rows.append(dict(row, file=record['_id'], group=group, split=split))
            stats['clean_' + row['kind'] + '_' + split] += 1
        if not issues and record.get('s3', {}).get('img_key'):
            docs.append({'id': record['_id'], 'group': group, 'split': split, 'image_key': record['s3']['img_key'], 'labels': labels})
            stats['layout_' + split] += 1
    with (args.output / 'clean.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['file','group','split','kind','raw','clean','ocr_indices'])
        writer.writeheader(); writer.writerows(clean_rows)
    (args.output / 'layout.json').write_text(json.dumps(docs))
    (args.output / 'splits.json').write_text(json.dumps(splits))
    report = {'source_sha256': hashlib.sha256(args.records.read_bytes()).hexdigest(), 'export': metadata,
              'exported_records': len(all_records), 'reviewed_records': len(records), 'counts': stats,
              'layout_exclusion_reasons': reasons, 'supervision': 'automatic alignment of business review, not manual BIO labels'}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
