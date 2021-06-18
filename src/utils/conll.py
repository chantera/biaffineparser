import os
import subprocess
import sys


def read_conll(file):
    with open(file) as f:
        yield from parse_conll(f)


def parse_conll(lines):
    def _create_root():
        token = {
            "id": 0,
            "form": "<ROOT>",
            "lemma": "<ROOT>",
            "cpostag": "ROOT",
            "postag": "ROOT",
            "feats": "_",
            "head": 0,
            "deprel": "root",
        }
        return token

    tokens = [_create_root()]
    for line in lines:
        line = line.strip()
        if not line:
            if len(tokens) > 1:
                yield tokens
                tokens = [_create_root()]
        elif line.startswith("#"):
            continue
        else:
            cols = line.split("\t")
            token = {
                "id": int(cols[0]),
                "form": cols[1],
                "lemma": cols[2],
                "cpostag": cols[3],
                "postag": cols[4],
                "feats": cols[5],
                "head": int(cols[6]),
                "deprel": cols[7],
            }
            tokens.append(token)
    if len(tokens) > 1:
        yield tokens


def write_conll(file, docs):
    with open(file, "w") as f:
        dump_conll(docs, f)


def dump_conll(docs, writer=sys.stdout):
    attrs = ["id", "form", "lemma", "cpostag", "postag", "feats", "head", "deprel"]
    for tokens in docs:
        for token in tokens:
            if token["id"] == 0:
                continue
            cols = map(lambda v: str(v) if v is not None else "_", (token.get(k) for k in attrs))
            writer.write("\t".join(cols) + "\n")
        writer.write("\n")
    writer.flush()


_EVAL_SCRIPT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "eval.pl")


def evaluate(gold_file, system_file, verbose=False):
    command = ["/usr/bin/perl", _EVAL_SCRIPT, "-g", gold_file, "-s", system_file]
    if not verbose:
        command.append("-q")
    option = {}
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **option)
    output = (p.stdout if p.returncode == 0 else p.stderr).decode("utf-8")
    if p.returncode != 0:
        error = p.stderr.decode("utf-8")
        raise RuntimeError("code={!r}, message={!r}".format(p.returncode, error))
    output = p.stdout.decode("utf-8")
    scores = [float(line.rsplit(" ", 2)[-2]) for line in output.split("\n", 2)[:2]]
    return dict(LAS=scores[0], UAS=scores[1], raw=output)
