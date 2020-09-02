from bert_score import BERTScorer
# import files2rouge
import numpy as np
# from rouge import Rouge, FilesRouge
import matplotlib.pyplot as plt

NUM_SAMPLES = 5
DIR = '/Users/alexgaskell/GoogleDrive/Documents/Imperial/Individual_project/datasets/cnn_dm/rouge_tests/'


def main():
    P, R, F1 = run_bert_score()
    r_scores = run_rouge()
    

    print(F1.tolist(), r_scores[:,0])
    plt.scatter(F1.tolist(), r_scores[:,0])
    plt.show()



def run_bert_score():
    # hyp_path = list(sorted(glob('summaries/*')))[-1]
    # hyps = [" " + x.rstrip() for x in open(hyp_path).readlines()]
    # refs = [" " + x.rstrip() for x in open(path.join(args.source_dir, 'test.source')).readlines()]

    # cands = [line.strip() for line in open(DIR + "small.hypo").readlines()[:NUM_SAMPLES]]
    # refs = [line.strip() for line in open(DIR + "small.target").readlines()[:NUM_SAMPLES]]
    
    cands = ["Celtic beat Dundee 0-0 away at Den's Park on wednesday night. The win sees them extend their lead at the top of the Scottish Premiership. Gary Mackay-Steven set the Bhoys on their way with a cool first-half finish. Virgil van Dijk scored a stunning free-kick in the second-half. Jim McAlister grabbed a late consolation for the home side on 00 minutes."]
    refs = ["Virgil Van Dijk is absent from the shortlists for this season's player of the year awards, Celtic will surely fire off a stiff missive seeking clarification on the reasons. The prospect of a treble evaporated last Sunday, an afternoon Ronny Deila now describes as the worst of his career."]

    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score(cands, refs)
    # scorer.plot_example(cands[0], refs[0], fname="summaries/bertscore_plot.jpg")
    # print(P)
    # print(R)
    # print(F1)
    # print('Avg:', F1.mean().item())
    # print('Best:', F1.argmax().item())
    # print('Worst:', F1.argmin().item())
    return P, R, F1

def run_rouge(variant='lines'):
    
    # # outfiles = list(sorted(glob('summaries/*')))
    # # examples = [" " + x.rstrip() for x in open(outfiles[-1]).readlines()]

    hyps = DIR + "small.hypo.tokenized"
    refs = DIR + "small.target.tokenized"

    if variant == 'official':
        files2rouge.run(hyps, refs) #, saveto='rouge_output.txt')

    elif variant == 'files':
        scores = FilesRouge().get_scores(hyps, refs, avg=True)
        print(score)

    else:
        cands = [line.strip() for line in open(hyps).readlines()[:NUM_SAMPLES]]
        refs = [line.strip() for line in open(refs).readlines()[:NUM_SAMPLES]]

        scrs = [Rouge().get_scores(c, r) for c,r in zip(cands, refs)]
        scores = np.array([[s[0]['rouge-1']['f'], s[0]['rouge-2']['f'], s[0]['rouge-l']['f']]  for s in scrs])

        # print(np.mean(scores, axis=0))
        # print(scores)
        return scores

if __name__ == '__main__':
    # main()
    print(run_bert_score()[-1].item())