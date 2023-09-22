import os
import json
import numpy as np
import collections

import networkx as nx
import matplotlib.pyplot as plt

# setup file path
root_dir = '/home/wangye/wangye2/wangye/TIP2021-erase'

anno_dir = root_dir
os.makedirs(anno_dir, exist_ok=True)

# input files
sent2srl_file = os.path.join(anno_dir, 'scripts/sent2srl_didemo_test.json')

# output files
sent2rg_file = os.path.join(anno_dir, 'sent2rolegraph_didemo_test.json')
sent2rga_file = os.path.join(anno_dir, 'sent2rolegraph_didemo_test.augment.json')

# convert sentence to role graph
sent2srl = json.load(open(sent2srl_file))


def create_role_graph_data(srl_data):
    words = srl_data['words']
    verb_items = srl_data['verbs']

    graph_nodes = {}
    graph_edges = []

    root_name = 'ROOT'
    graph_nodes[root_name] = {'words': words, 'spans': list(range(0, len(words))), 'role': 'ROOT'}

    # parse all verb_items
    phrase_items = []
    for i, verb_item in enumerate(verb_items):
        tags = verb_item['tags']
        tag2idxs = {}
        tagname_counter = {}  # multiple args of the same role
        for t, tag in enumerate(tags):
            if tag == 'O':
                continue
            if t > 0 and tag[0] != 'B':
                # deal with some parsing mistakes, e.g. (B-ARG0, O-ARG1)
                # change it into (B-ARG0, B-ARG1)
                if tag[2:] != tags[t - 1][2:]:
                    tag = 'B' + tag[1:]
            tagname = tag[2:]
            if tag[0] == 'B':
                if tagname not in tagname_counter:
                    tagname_counter[tagname] = 1
                else:
                    tagname_counter[tagname] += 1
            new_tagname = '%s:%d' % (tagname, tagname_counter[tagname])
            tag2idxs.setdefault(new_tagname, [])
            tag2idxs[new_tagname].append(t)
        if len(tagname_counter) > 1 and 'V' in tagname_counter and tagname_counter['V'] == 1:
            phrase_items.append(tag2idxs)

    node_idx = 1
    spanrole2nodename = {}
    for i, phrase_item in enumerate(phrase_items):
        # add verb node to graph
        tagname = 'V:1'
        role = 'V'
        spans = phrase_item[tagname]
        spanrole = '-'.join([str(x) for x in spans] + [role])
        if spanrole in spanrole2nodename:
            continue
        node_name = str(node_idx)
        tag_words = [words[idx] for idx in spans]
        graph_nodes[node_name] = {
            'role': role, 'spans': spans, 'words': tag_words,
        }
        spanrole2nodename[spanrole] = node_name
        verb_node_name = node_name
        node_idx += 1

        # add arg nodes and edges of the verb node
        for tagname, spans in phrase_item.items():
            role = tagname.split(':')[0]
            if role != 'V':
                spanrole = '-'.join([str(x) for x in spans] + [role])
                if spanrole in spanrole2nodename:
                    node_name = str(spanrole2nodename[spanrole])
                else:
                    # add new node or duplicate a node with a different role
                    node_name = str(node_idx)
                    tag_words = [words[idx] for idx in spans]
                    graph_nodes[node_name] = {
                        'role': role, 'spans': spans, 'words': tag_words,
                    }
                    spanrole2nodename[spanrole] = node_name
                    node_idx += 1
                # add edge
                graph_edges.append((verb_node_name, node_name, role))

    return graph_nodes, graph_edges


sent2graph = {}
for sent, srl in sent2srl.items():
    try:
        graph_nodes, graph_edges = create_role_graph_data(srl)
        sent2graph[sent] = (graph_nodes, graph_edges)
    except:
        print(sent)

json.dump(sent2graph, open(sent2rg_file, 'w'))

n = 0
for sent, graph in sent2graph.items():
  if len(graph[0]) == 1:
    n += 1
#     print(sent)
print('#sents without non-root nodes:', n)

import spacy

# ! python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

for sent, graph in sent2graph.items():
    nodes, edges = graph
    node_idx = len(nodes)

    # add noun and verb word node if no noun and no noun phrases
    if len(nodes) == 1:
        doc = nlp(sent)
        # print(len(doc))
        # print(len(nodes['ROOT']['words']))
        # print(doc.noun_chunks)
        # print(nodes['ROOT']['words'])
        # assert len(doc) == len(nodes['ROOT']['words']), sent

        # add noun nodes
        for w in doc.noun_chunks:
            node_name = str(node_idx)
            nodes[node_name] = {
                'role': 'NOUN', 'spans': np.arange(w.start, w.end).tolist()
            }
            nodes[node_name]['words'] = [doc[j].text for j in nodes[node_name]['spans']]
            node_idx += 1
        if len(nodes) == 1:
            for w in doc:
                node_name = str(node_idx)
                if w.tag_.startswith('NN'):
                    nodes[node_name] = {
                        'role': 'NOUN', 'spans': [w.i], 'words': [w.text],
                    }
                    node_idx += 1

        # add verb nodes
        for w in doc:
            node_name = str(node_idx)
            if w.tag_.startswith('VB'):
                nodes[node_name] = {
                    'role': 'V', 'spans': [w.i], 'words': [w.text],
                }
                node_idx += 1

    sent2graph[sent] = (nodes, edges)

print(len(sent2graph))

json.dump(sent2graph, open(sent2rga_file, 'w'))

role_types = collections.Counter()
for sent, graph in sent2graph.items():
  nodes = graph[0]
  for k, v in nodes.items():
    role_types[v['role']] += 1
print(len(role_types))

role_types.most_common()

# noun per sent
nouns_per_sent = []
for sent, graph in sent2graph.items():
  n_nouns = 0
  for node_id, node in graph[0].items():
    if node['role'] != 'ROOT' and node['role'] != 'V':
      n_nouns += 1
  if n_nouns == 0:
    print(sent)
  nouns_per_sent.append(n_nouns)
nouns_per_sent = np.array(nouns_per_sent)
print(np.sum(nouns_per_sent == 0), np.min(nouns_per_sent), np.mean(nouns_per_sent), np.max(nouns_per_sent),
     np.percentile(nouns_per_sent, 90), np.percentile(nouns_per_sent, 95))


# verb per sent
verbs_per_sent = []
for sent, graph in sent2graph.items():
  n_verbs = 0
  for node_id, node in graph[0].items():
    if node['role'] == 'V':
      n_verbs += 1
  verbs_per_sent.append(n_verbs)
verbs_per_sent = np.array(verbs_per_sent)
print(np.sum(verbs_per_sent == 0), np.min(verbs_per_sent), np.mean(verbs_per_sent), np.max(verbs_per_sent),
     np.percentile(verbs_per_sent, 90), np.percentile(verbs_per_sent, 96))

print(sent2graph['There are a lot of people in an opened court area and they look to be mostly elderly people playing some kind of sport'])