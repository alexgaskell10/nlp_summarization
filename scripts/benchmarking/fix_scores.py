import json

infile = '/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis/eval_output_raw.txt'
path = '/vol/bitbucket/aeg19/datasets/cnn_dm/adversarial/permuted.txt'
ref = '../../../datasets/cnn_dm/pegasus/test.hypo'
scores = [eval(line) for line in open(infile, 'r')]

corr = [s['rouge1'] for s in scores if s['hyps_path'] == path]
refs = [s['rouge1'] for s in scores if s['hyps_path'] == ref]
[print(c, r) for c,r in zip(corr[0][:10], refs[0][:10])]

# full = [line for line in scores if 'rouge1' in line.keys()]
# half = [line for line in scores if 'rouge1' not in line.keys()]

# for line in half:
#     index = [i for i,l in enumerate(full) if l['hyps_path'] == line['hyps_path']][0]
#     full[index]['bertscore'] = line['bertscore']

# outfile = '/vol/bitbucket/aeg19/datasets/bart-pubmed/analysis/eval_output_raw_new.txt'
# for line in full:
#     with open(outfile, 'a+') as f:
#         json.dump(line, f)
#         f.write('\n')
