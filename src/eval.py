import os
import subprocess
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import teras


class Evaluator(teras.training.event.Listener):
    name = "parsing_evaluator"

    def __init__(self, parser, rel_map, gold_file, logger=None, **kwargs):
        super().__init__(**kwargs)
        self._parser = parser
        self._rel_map = rel_map
        self._logger = logger \
            if logger is not None else teras.utils.logging.getLogger('teras')
        self.set_gold(gold_file)

    def set_gold(self, gold_file):
        self._gold_file = os.path.abspath(os.path.expanduser(gold_file))
        self.reset()

    def reset(self):
        self._parsed = {'records': [], 'UAS': None, 'LAS': None}

    def append(self, sentences, pred_heads, pred_rels, gold_heads, gold_rels):
        records = []
        rel_map = self._rel_map
        for i, (tokens, p_heads, p_rels, t_heads, t_rels) in enumerate(
                zip(sentences, pred_heads, pred_rels, gold_heads, gold_rels)):
            n = len(tokens)
            assert n == len(p_heads) == len(t_heads)
            records.append({
                'sentence': tokens[1:],
                'pred_heads': p_heads[1:],
                'pred_rels':
                [rel_map.lookup(rel if rel > 0 else 0) for rel in p_rels[1:]],
                'true_heads': t_heads[1:],
                'true_rels':
                [rel_map.lookup(rel if rel > 0 else 0) for rel in t_rels[1:]],
                'head_errors':
                np.where(p_heads[1:] != t_heads[1:])[0].astype(np.int32) + 1,
                'rel_errors':
                np.where(p_rels[1:] != t_rels[1:])[0].astype(np.int32) + 1,
            })
        self._parsed['records'].extend(records)

    def report(self, show_details=False):
        sentences, heads, rels = zip(
            *[(r['sentence'], r['pred_heads'], r['pred_rels'])
              for r in self._parsed['records']])
        with NamedTemporaryFile(mode='w') as f:
            write_conll(f, sentences, heads, rels)
            result = exec_eval(f.name, self._gold_file, show_details)
            if result['code'] != 0:  # retry
                with NamedTemporaryFile(mode='w') as gold_f:
                    write_conll(gold_f, sentences)
                    result = exec_eval(f.name, gold_f.name, show_details)
        if result['code'] == 0:
            self._parsed['UAS'] = result['UAS']
            self._parsed['LAS'] = result['LAS']
            message = "[evaluation]\n{}".format(result['raw'].rstrip())
        else:
            message = "[evaluation] ERROR({}): {}".format(
                result['code'], result['raw'].rstrip())
        self._logger.info(message)

    def on_batch_begin(self, data):
        pass

    def on_batch_end(self, data):
        if data['train']:
            return
        sentences = data['xs'][-1]
        parsed = self._parser.parse(*data['xs'][:-1], use_cache=True)
        pred_heads, pred_rels, *_ = zip(*parsed)
        gold_heads, gold_rels, *_ = zip(*data['ts'])
        self.append(sentences, pred_heads, pred_rels, gold_heads, gold_rels)

    def on_epoch_validate_begin(self, data):
        self.reset()

    def on_epoch_validate_end(self, data):
        self.report(show_details=False)

    @property
    def result(self):
        return self._parsed['records']


def write_conll(writer, sentences, heads=None, deprels=None):
    for i, tokens in enumerate(sentences):
        for j, token in enumerate(tokens):
            line = '\t'.join([
                str(token['id']),
                token['form'],
                token['lemma'],
                token['cpostag'],
                token['postag'],
                token['feats'],
                str(heads[i][j]) if heads is not None else str(token['head']),
                deprels[i][j] if deprels is not None else token['deprel'],
                token['phead'],
                token['pdeprel'],
            ])
            writer.write(line + '\n')
        writer.write('\n')
    writer.flush()


EVAL_SCRIPT = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'common', 'eval.pl')


def exec_eval(parsed_file, gold_file, show_details=False):
    command = ['/usr/bin/perl', EVAL_SCRIPT,
               '-g', gold_file, '-s', parsed_file]
    if not show_details:
        command.append('-q')
    print("exec command: {}".format(' '.join(command)), file=sys.stderr)
    option = {}
    p = subprocess.run(command,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       **option)
    output = (p.stdout if p.returncode == 0 else p.stderr).decode('utf-8')
    result = {
        'code': p.returncode,
        'raw': output,
        'UAS': None,
        'LAS': None,
    }
    if p.returncode == 0:
        lines = output.split('\n', 2)[:2]
        result['LAS'], result['UAS'] = \
            [float(line.rsplit(' ', 2)[-2]) for line in lines]
    return result
