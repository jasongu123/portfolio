"""
Distillation project, stage B: fine-tune a small model on Claude Haiku's labels (the "student").
Pre-registered in distill_label.py's docstring (2026-10-02) before any label existed.

Student: ProsusAI/finbert (BERT-base pre-trained on financial text) with one classification head
per label (tone, guidance, vs_expectations, hedging), input = the first 512 tokens of the release's
narrative text. Data: the labelled releases in the sample's (shuffled) order; the first 2,000 are
split 1,600 train / 400 validation (early stopping on mean validation accuracy, at most 4 epochs),
the last 500 are the held-out test, used once, after training, for the agreement report.

Quality gate (held-out 500, student vs teacher):
  tone within one category >= 90% and exact >= 60%; guidance exact >= 75%;
  vs_expectations exact >= 70%; hedging exact >= 60%;
  and every field beats always predicting its most common training label.

  python3 distill_student.py --train      (Apple GPU via MPS; ~10-30 minutes)

VERSION 2 (pre-registered 2026-10-02 after version 1 FAILED on guidance and hedging, before any
version-2 label existed; T-2026-10-02-90):
  Diagnosis of v1: the student read only its first 512 tokens (~2,000 characters) while the teacher
  read 12,000; 44% of releases first mention outlook or guidance after character 2,000.
  Student v2: the same FinBERT, reading the same first 12,000 characters as the teacher, split into
  510-token chunks (at most 8); each chunk is encoded, the chunk [CLS] vectors are mean- and
  max-pooled and concatenated, then one head per label.
  Training data: all 2,467 v1 Haiku labels + 1,500 new Haiku labels (earnings_text_train2.csv);
  400 of them (random, seed 42) for early stopping (at most 4 epochs); the rest for training.
  Test: the FRESH 500-release set (earnings_text_test2.csv), disjoint from everything above, used
  once. Gate unchanged from v1 (student vs Haiku): tone within one >= 90% and exact >= 60%;
  guidance >= 75%; vs_expectations >= 70%; hedging >= 60%; each beats the most common label.
  Reported alongside (no decision): Sonnet 5.5 vs Haiku agreement on the same 500 (how far two
  strong models agree: a ceiling for the student), and student vs Sonnet.
  python3 distill_student.py --train-v2    (~1-2 hours on the Mac's GPU)

VERSION 3 (pre-registered 2026-10-02, T-2026-10-02-91; labels, teacher and gate in distill_label.py):
  the v2 chunked student trained on Claude Sonnet 5.5's version-3 labels of the 4,450-release pool
  (400 for early stopping), tested once on the fresh test3 set against Sonnet, with the gate that
  `distill_label.py --v3 --wait` fixed from the two teachers before training.
  python3 distill_student.py --train-v3    (~2.5 hours on the Mac's GPU)
Writes models/finbert_earnings/ and results/distill/student_report.txt.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from distill_label import FIELDS, LABELS, SAMPLE, TEXT_DIR  # noqa: E402

BASE_MODEL = 'ProsusAI/finbert'
MODEL_DIR = os.path.join(HERE, 'models', 'finbert_earnings')
OUT = os.path.join(HERE, 'results', 'distill')
N_TRAINVAL, N_TRAIN = 2000, 1600
MAX_LEN, BATCH, LR, EPOCHS, SEED = 512, 16, 2e-5, 4, 42
GATE = {'tone_within1': 0.90, 'tone': 0.60, 'guidance': 0.75, 'vs_expectations': 0.70, 'hedging': 0.60}


def load_data():
    order = pd.read_csv(SAMPLE)[['accession']]
    lab = pd.read_csv(LABELS)
    df = order.merge(lab, on='accession', how='inner')
    df['text'] = [open(os.path.join(TEXT_DIR, f'{a}.txt')).read() for a in df.accession]
    for k, v in FIELDS.items():
        df[k + '_y'] = df[k].map({c: i for i, c in enumerate(v)})
    return df.iloc[:N_TRAIN], df.iloc[N_TRAIN:N_TRAINVAL], df.iloc[N_TRAINVAL:]


def build_model():
    import torch
    from transformers import AutoModel

    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(BASE_MODEL)
            h = self.enc.config.hidden_size
            self.drop = torch.nn.Dropout(0.1)
            self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(h, len(v)) for k, v in FIELDS.items()})

        def forward(self, ids, mask):
            cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            cls = self.drop(cls)
            return {k: head(cls) for k, head in self.heads.items()}

    return Student()


def encode(tok, texts):
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']


def predict(model, tok, texts, device, batch=32):
    import torch
    model.eval()
    probs = {k: [] for k in FIELDS}
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            ids, mask = encode(tok, texts[i:i + batch])
            out = model(ids.to(device), mask.to(device))
            for k in FIELDS:
                probs[k].append(torch.softmax(out[k], -1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in probs.items()}


def accuracy(probs, df, fields=FIELDS):
    acc = {k: float((probs[k].argmax(1) == df[k + '_y'].values).mean()) for k in fields}
    acc['tone_within1'] = float((np.abs(probs['tone'].argmax(1) - df['tone_y'].values) <= 1).mean())
    return acc


def train():
    import torch
    from transformers import AutoTokenizer
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tr, va, te = load_data()
    print(f"train {len(tr)}, validation {len(va)}, held-out test {len(te)}; device {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ids, mask = encode(tok, tr.text)
    ys = {k: torch.tensor(tr[k + '_y'].values) for k in FIELDS}
    loss_fn = torch.nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(tr), BATCH):
            b = perm[i:i + BATCH]
            out = model(ids[b].to(device), mask[b].to(device))
            loss = sum(loss_fn(out[k], ys[k][b].to(device)) for k in FIELDS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        acc = accuracy(predict(model, tok, va.text.tolist(), device), va)
        score = float(np.mean([acc[k] for k in FIELDS]))
        print(f"  epoch {ep + 1}: train loss {tot / (len(tr) / BATCH):.3f}, validation " +
              ", ".join(f"{k} {v:.2f}" for k, v in acc.items()) + f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'student.pt'))
    tok.save_pretrained(MODEL_DIR)

    # held-out test: used once
    probs = predict(model, tok, te.text.tolist(), device)
    acc = accuracy(probs, te)
    lines = [f"Held-out test, {len(te)} releases never used in training or model selection (student vs Claude Haiku):"]
    ok_all = True
    for k in ['tone_within1'] + list(FIELDS):
        base = None if k == 'tone_within1' else float((te[k + '_y'] == tr[k + '_y'].mode()[0]).mean())
        need = GATE[k]
        ok = acc[k] >= need and (base is None or acc[k] > base)
        ok_all &= ok
        lines.append(f"  {k:16s} agreement {acc[k]:.1%} (need {need:.0%}" + (f"; most-common-label baseline {base:.1%}" if base is not None else '') +
                     f") {'ok' if ok else 'FAIL'}")
    for k, v in FIELDS.items():
        cm = pd.crosstab(pd.Series([v[i] for i in te[k + '_y']], name='teacher'),
                         pd.Series([v[i] for i in probs[k].argmax(1)], name='student'))
        lines.append(f"\n{k} confusion (rows teacher, columns student):\n{cm.to_string()}")
    lines.append(f"\nSTUDENT QUALITY GATE: {'PASS' if ok_all else 'FAIL'}")
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(OUT, 'student_report.txt'), 'w').write(report + '\n')


CHUNK_TOKENS, MAX_CHUNKS, DOC_BATCH = 510, 8, 4
V2_TRAIN = [os.path.join(os.path.dirname(LABELS), 'earnings_labels_haiku.csv'),
            os.path.join(os.path.dirname(LABELS), 'earnings_labels_haiku_train2.csv')]
V2_TEST = os.path.join(os.path.dirname(LABELS), 'earnings_labels_haiku_test2.csv')
V2_SONNET = os.path.join(os.path.dirname(LABELS), 'earnings_labels_sonnet_test2.csv')
MODEL_DIR_V2 = os.path.join(HERE, 'models', 'finbert_earnings_v2')


def _with_text(df, fields=FIELDS):
    from distill_label import INPUT_CHARS
    df = df.copy()
    df['text'] = [open(os.path.join(TEXT_DIR, f'{a}.txt')).read()[:INPUT_CHARS] for a in df.accession]
    for k, v in fields.items():
        df[k + '_y'] = df[k].map({c: i for i, c in enumerate(v)})
    return df


def chunk_ids(tok, text):
    """[CLS] + up to 510 tokens + [SEP] per chunk, at most MAX_CHUNKS chunks, covering the text in order."""
    ids = tok(text, add_special_tokens=False)['input_ids']
    chunks = [ids[i:i + CHUNK_TOKENS] for i in range(0, max(len(ids), 1), CHUNK_TOKENS)][:MAX_CHUNKS]
    return [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]


def build_model_v2(fields=FIELDS):
    import torch
    from transformers import AutoModel

    class ChunkStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(BASE_MODEL)
            # recompute activations in the backward pass: 4 releases x 8 chunks x 512 tokens otherwise
            # needs ~20 GB of GPU memory on a 24 GB Mac and swaps (2026-10-02)
            self.enc.gradient_checkpointing_enable()
            h = self.enc.config.hidden_size
            self.drop = torch.nn.Dropout(0.1)
            self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(2 * h, len(v)) for k, v in fields.items()})

        def forward(self, ids, mask, chunk_valid, n_docs):
            cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
            k = ids.shape[0] // n_docs
            cls = cls.view(n_docs, k, -1)
            v = chunk_valid.view(n_docs, k, 1).to(cls.dtype)
            mean = (cls * v).sum(1) / v.sum(1).clamp(min=1.0)
            mx = cls.masked_fill(v == 0, -1e4).max(1).values
            x = self.drop(torch.cat([mean, mx], dim=1))
            return {k: head(x) for k, head in self.heads.items()}

    return ChunkStudent()


def _batch(tok, chunk_lists, device, n_docs=None):
    """Fixed shape (n_docs x MAX_CHUNKS, 512) whatever the releases: the MPS allocator keeps a buffer
    per tensor shape, and variable shapes grew the first v2 run to 27 GB (2026-10-02)."""
    import torch
    n_docs = n_docs or len(chunk_lists)
    width = CHUNK_TOKENS + 2
    k = max(len(c) for c in chunk_lists)                 # 1..MAX_CHUNKS: at most 8 distinct shapes
    ids = torch.full((n_docs * k, width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((n_docs * k, width), dtype=torch.long)
    valid = torch.zeros(n_docs * k, dtype=torch.long)
    for d, chunks in enumerate(chunk_lists):
        for j, x in enumerate(chunks[:k]):
            r = d * k + j
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
            valid[r] = 1
    for d in range(len(chunk_lists), n_docs):            # padding documents (last partial batch)
        r = d * k
        ids[r, :2] = torch.tensor([tok.cls_token_id, tok.sep_token_id])
        mask[r, :2] = 1
        valid[r] = 1
    return ids.to(device), mask.to(device), valid.to(device), n_docs


def predict_v2(model, tok, chunk_lists, device, fields=FIELDS):
    import torch
    model.eval()
    probs = {k: [] for k in fields}
    with torch.no_grad():
        for i in range(0, len(chunk_lists), DOC_BATCH):
            part = chunk_lists[i:i + DOC_BATCH]
            out = model(*_batch(tok, part, device, n_docs=DOC_BATCH))
            for k in fields:
                probs[k].append(torch.softmax(out[k], -1).cpu().numpy()[:len(part)])
        if device == 'mps':
            torch.mps.empty_cache()
    return {k: np.concatenate(v) for k, v in probs.items()}


def train_v2():
    import torch
    from transformers import AutoTokenizer
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pool = _with_text(pd.concat([pd.read_csv(f) for f in V2_TRAIN]).drop_duplicates('accession').reset_index(drop=True))
    test = _with_text(pd.read_csv(V2_TEST))
    assert not set(test.accession) & set(pool.accession), "test set overlaps training data"
    va = pool.sample(400, random_state=SEED)
    tr = pool.drop(va.index)
    assert len(tr) + len(va) == len(pool) and not set(tr.accession) & set(va.accession)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ch = {name: [chunk_ids(tok, t) for t in df.text] for name, df in (('tr', tr), ('va', va), ('te', test))}
    print(f"train {len(tr)}, validation {len(va)}, fresh test {len(test)}; mean chunks per release "
          f"{np.mean([len(c) for c in ch['tr']]):.1f}; device {device}", flush=True)
    model = build_model_v2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    ys = {k: torch.tensor(tr[k + '_y'].values) for k in FIELDS}
    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        # releases of similar length share a batch (less padding); batch order is shuffled each epoch
        n_ch = np.array([len(c) for c in ch['tr']])
        order = np.lexsort((np.random.permutation(len(tr)), n_ch))
        batches = [order[i:i + DOC_BATCH] for i in range(0, len(order), DOC_BATCH)]
        np.random.shuffle(batches)
        t0, tot = time.time(), 0.0
        steps = len(batches)
        for step, b in enumerate(batches):
            if len(b) < DOC_BATCH:                       # keep every batch the same size
                b = np.concatenate([b, order[:DOC_BATCH - len(b)]])
            out = model(*_batch(tok, [ch['tr'][j] for j in b], device))
            loss = sum(loss_fn(out[k], ys[k][b].to(device)) for k in FIELDS)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            if device == 'mps' and step % 50 == 0:
                torch.mps.empty_cache()
            if step % 100 == 0:
                mem = torch.mps.driver_allocated_memory() / 1e9 if device == 'mps' else 0.0
                print(f"    epoch {ep + 1} step {step}/{steps} ({(time.time() - t0) / 60:.1f} min, GPU memory {mem:.1f} GB)", flush=True)
        acc = accuracy(predict_v2(model, tok, ch['va'], device), va)
        score = float(np.mean([acc[k] for k in FIELDS]))
        print(f"  epoch {ep + 1}: train loss {tot / (len(tr) / DOC_BATCH):.3f}, validation " +
              ", ".join(f"{k} {v:.2f}" for k, v in acc.items()) + f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    os.makedirs(MODEL_DIR_V2, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR_V2, 'student.pt'))
    tok.save_pretrained(MODEL_DIR_V2)

    probs = predict_v2(model, tok, ch['te'], device)              # the fresh test set, used once
    acc = accuracy(probs, test)
    lines = [f"Version 2 fresh test: {len(test)} releases (student vs Claude Haiku 4.5):"]
    ok_all = True
    for k in ['tone_within1'] + list(FIELDS):
        base = None if k == 'tone_within1' else float((test[k + '_y'] == tr[k + '_y'].mode()[0]).mean())
        ok = acc[k] >= GATE[k] and (base is None or acc[k] > base)
        ok_all &= ok
        lines.append(f"  {k:16s} agreement {acc[k]:.1%} (need {GATE[k]:.0%}" +
                     (f"; most-common-label baseline {base:.1%}" if base is not None else '') + f") {'ok' if ok else 'FAIL'}")
    if os.path.exists(V2_SONNET):
        son = pd.read_csv(V2_SONNET).set_index('accession')
        m = test.set_index('accession').join(son, rsuffix='_sonnet', how='inner')
        lines.append(f"\nTeacher agreement (no decision), {len(m)} releases labelled by both:")
        stu = {k: pd.Series([FIELDS[k][i] for i in probs[k].argmax(1)], index=test.accession) for k in FIELDS}
        for k in FIELDS:
            hs = (m[k] == m[k + '_sonnet']).mean()
            ss = (stu[k].reindex(m.index) == m[k + '_sonnet']).mean()
            lines.append(f"  {k:16s} Sonnet 5.5 vs Haiku {hs:.1%} | student vs Sonnet 5.5 {ss:.1%}")
    for k, v in FIELDS.items():
        cm = pd.crosstab(pd.Series([v[i] for i in test[k + '_y']], name='teacher'),
                         pd.Series([v[i] for i in probs[k].argmax(1)], name='student'))
        lines.append(f"\n{k} confusion (rows teacher, columns student):\n{cm.to_string()}")
    lines.append(f"\nSTUDENT V2 QUALITY GATE: {'PASS' if ok_all else 'FAIL'}")
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(OUT, 'student_v2_report.txt'), 'w').write(report + '\n')


V3_POOL = os.path.join(os.path.dirname(LABELS), 'earnings_labels3_sonnet_train.csv')
V3_TEST = os.path.join(os.path.dirname(LABELS), 'earnings_labels3_sonnet_test.csv')
V3_HAIKU = os.path.join(os.path.dirname(LABELS), 'earnings_labels3_haiku_test.csv')
MODEL_DIR_V3 = os.path.join(HERE, 'models', 'finbert_earnings_v3')


def fit_chunked(tr, va, ch_tr, ch_va, fields, tok, device, epochs=EPOCHS):
    """The v2 training loop for any label set: fixed-shape length-bucketed batches, early stopping on
    mean validation accuracy. Returns the best model (by validation)."""
    import torch
    model = build_model_v2(fields).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    ys = {k: torch.tensor(tr[k + '_y'].values) for k in fields}
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        n_ch = np.array([len(c) for c in ch_tr])
        order = np.lexsort((np.random.permutation(len(tr)), n_ch))
        batches = [order[i:i + DOC_BATCH] for i in range(0, len(order), DOC_BATCH)]
        np.random.shuffle(batches)
        t0, tot, steps = time.time(), 0.0, len(batches)
        for step, b in enumerate(batches):
            if len(b) < DOC_BATCH:
                b = np.concatenate([b, order[:DOC_BATCH - len(b)]])
            out = model(*_batch(tok, [ch_tr[j] for j in b], device))
            loss = sum(loss_fn(out[k], ys[k][b].to(device)) for k in fields)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            if device == 'mps' and step % 50 == 0:
                torch.mps.empty_cache()
            if step % 100 == 0:
                mem = torch.mps.driver_allocated_memory() / 1e9 if device == 'mps' else 0.0
                print(f"    epoch {ep + 1} step {step}/{steps} ({(time.time() - t0) / 60:.1f} min, GPU memory {mem:.1f} GB)", flush=True)
        acc = accuracy(predict_v2(model, tok, ch_va, device, fields), va, fields)
        score = float(np.mean([acc[k] for k in fields]))
        print(f"  epoch {ep + 1}: train loss {tot / max(steps, 1):.3f}, validation " +
              ", ".join(f"{k} {v:.2f}" for k, v in acc.items()) + f" ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def judge_v3(acc, baselines, gates):
    """Apply the pre-registered v3 gate (distill_label.py docstring). Returns (lines, passed)."""
    from distill_label import FIELDS_V3
    lines, ok_all = [], gates['any_guidance_included']
    for k in FIELDS_V3:
        g = gates['fields'][k]
        if not g['included']:
            lines.append(f"  {k:18s} student vs Sonnet {acc[k]:.1%}  (excluded: teachers agree only "
                         f"{g['teacher_agreement']:.1%})")
            continue
        need = max(g['need'], baselines[k] + MARGIN_V3)
        ok = acc[k] >= need
        line = (f"  {k:18s} student vs Sonnet {acc[k]:.1%} (need {need:.1%}: Haiku vs Sonnet "
                f"{g['teacher_agreement']:.1%} - 3, baseline {baselines[k]:.1%} + 3)")
        if k == 'tone':
            ok &= acc['tone_within1'] >= g['need_within1']
            line += f"; within one {acc['tone_within1']:.1%} (need {g['need_within1']:.0%})"
        ok_all &= ok
        lines.append(line + f" {'ok' if ok else 'FAIL'}")
    if not gates['any_guidance_included']:
        lines.append("  no guidance field is reproducible between the teachers -> FAIL")
    return lines, bool(ok_all)


MARGIN_V3 = 0.03


def train_v3():
    import json
    import torch
    from transformers import AutoTokenizer
    from distill_label import FIELDS_V3, GATES_V3
    if not os.path.exists(GATES_V3):
        sys.exit(f"{GATES_V3} missing: run `python3 distill_label.py --v3 --wait` first (it fixes the gate)")
    gates = json.load(open(GATES_V3))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    pool = _with_text(pd.read_csv(V3_POOL).drop_duplicates('accession').reset_index(drop=True), FIELDS_V3)
    test = _with_text(pd.read_csv(V3_TEST), FIELDS_V3)
    assert not set(test.accession) & set(pool.accession), "test set overlaps training data"
    va = pool.sample(400, random_state=SEED)
    tr = pool.drop(va.index)
    assert len(tr) + len(va) == len(pool) and not set(tr.accession) & set(va.accession)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    ch = {name: [chunk_ids(tok, t) for t in df.text] for name, df in (('tr', tr), ('va', va), ('te', test))}
    print(f"train {len(tr)}, validation {len(va)}, fresh test {len(test)} (Sonnet 5.5 labels); device {device}", flush=True)
    print("label shares (train): " + "; ".join(
        f"{k} " + ", ".join(f"{c} {p:.0%}" for c, p in tr[k].value_counts(normalize=True).items()) for k in FIELDS_V3), flush=True)
    model = fit_chunked(tr, va, ch['tr'], ch['va'], FIELDS_V3, tok, device)
    os.makedirs(MODEL_DIR_V3, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR_V3, 'student.pt'))
    tok.save_pretrained(MODEL_DIR_V3)

    probs = predict_v2(model, tok, ch['te'], device, FIELDS_V3)    # the fresh test set, used once
    acc = accuracy(probs, test, FIELDS_V3)
    baselines = {k: float((test[k + '_y'] == tr[k + '_y'].mode()[0]).mean()) for k in FIELDS_V3}
    lines, passed = judge_v3(acc, baselines, gates)
    lines = [f"Version 3 fresh test: {len(test)} releases (student vs Claude Sonnet 5.5); gate fixed "
             f"{gates['fixed_at']} from the teachers alone:"] + lines
    if os.path.exists(V3_HAIKU):
        h = pd.read_csv(V3_HAIKU).set_index('accession')
        stu = {k: pd.Series([FIELDS_V3[k][i] for i in probs[k].argmax(1)], index=test.accession) for k in FIELDS_V3}
        m = test.set_index('accession').join(h, rsuffix='_haiku', how='inner')
        lines.append(f"\nStudent vs Haiku 4.5 (no decision), {len(m)} releases:")
        for k in FIELDS_V3:
            lines.append(f"  {k:18s} {(stu[k].reindex(m.index) == m[k + '_haiku']).mean():.1%}")
    for k, v in FIELDS_V3.items():
        cm = pd.crosstab(pd.Series([v[i] for i in test[k + '_y']], name='teacher'),
                         pd.Series([v[i] for i in probs[k].argmax(1)], name='student'))
        lines.append(f"\n{k} confusion (rows Sonnet, columns student):\n{cm.to_string()}")
    lines.append(f"\nSTUDENT V3 QUALITY GATE: {'PASS' if passed else 'FAIL'}")
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(OUT, 'student_v3_report.txt'), 'w').write(report + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--train-v2', action='store_true')
    ap.add_argument('--train-v3', action='store_true')
    a = ap.parse_args()
    if a.train:
        train()
    if a.train_v2:
        train_v2()
    if a.train_v3:
        train_v3()
