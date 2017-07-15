DEVELOP = True


def read_conll(file):
    root = {
        'id': 0,
        'form': "<ROOT>",
        # 'lemma': "<ROOT>",
        # 'cpostag': "ROOT",
        'postag':  "ROOT",
        # 'feats':   "_",
        'head': 0,
        'deprel':  "root",
        # 'phead':   "_",
        # 'pdeprel': "_",
    }
    sentences = []
    tokens = []
    for line in open(file, 'r'):
        line = line.strip()
        if not line and tokens:
            sentences.append([root] + tokens)
            tokens = []
        elif line.startswith('#'):
            continue
        else:
            cols = line.split("\t")
            row = {
                'id': int(cols[0]),
                'form': cols[1],
                # 'lemma': cols[2],
                # 'cpostag': cols[3],
                'postag': cols[4],
                # 'feats': cols[5],
                'head': int(cols[6]),
                'deprel': cols[7],
                # 'phead': int(cols[8]),
                # 'pdeprel': cols[9],
            }
            tokens.append(row)
    if tokens:
        sentences.append([root] + tokens)
    return sentences
