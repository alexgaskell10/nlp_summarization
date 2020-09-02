import torch
import pickle
import fairseq

print('Ready')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn', force_reload=False)

# PATH = '/Users/alexgaskell/GoogleDrive/Documents/Imperial/Individual_project/test_save_bert.pt'
# # PATH = '/content/drive/My Drive/Documents/Imperial/Individual_project/test_save.pt'
# torch.save(bart.state_dict(), PATH)
# PATH = '/Users/alexgaskell/GoogleDrive/Documents/Imperial/Individual_project/'
# PATH = '/content/drive/My Drive/Documents/Imperial/Individual_project/'
PATH = '/Users/alexgaskell/GoogleDrive/Documents/Imperial/Individual_project/cnn_dm/'

# bart.cuda()
# bart.half()
bart.eval()
count = 1
bsz = 1

# with open(PATH + 'test.source') as source, open(PATH + 'test.hypo', 'w') as fout:
with open(PATH + 'partial/test.source.sample-1000') as source, open(PATH + 'partial/test.hypo.sample-1000', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
