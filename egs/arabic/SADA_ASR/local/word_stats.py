import matplotlib.pyplot as plt


def word_stats(file):
    stats = {}
    with open(file, 'r') as f:
        modules = f.read().split('\n\n')
        word_stats = [module for module in modules if module.startswith('PER-WORD STATS:')]
        word_stats = word_stats[0].split('\n')[1:-1]
        for line in word_stats:
            word, corr, tot_errs, _, _ = line.split()
            key, corr, tot_errs = len(word), int(corr), int(tot_errs)
            if key in stats:
                stats[key]['count'] += 1
                stats[key]['corr'] += corr
                stats[key]['tot_errs'] += tot_errs
            else:
                stats[key] = {}
                stats[key]['count'] = 1
                stats[key]['corr'] = corr
                stats[key]['tot_errs'] = tot_errs
    for k in stats:
        stats[k]['acc'] = stats[k]['corr'] / (stats[k]['corr'] + stats[k]['tot_errs'])
    return stats


def plot_stats(stats):
    plt.switch_backend('agg')
    stats = sorted(stats.items())
    length, acc = [], []
    for k, v in stats:
        length.append(k)
        acc.append(v['acc'])
    plt.xlabel('word length')
    plt.ylabel('p correct')
    plt.xticks(range(1, 21, 1))
    plt.bar(length, acc)
    plt.savefig('./correct.jpg')
    
def plot_count(stats):
    plt.switch_backend('agg')
    stats = sorted(stats.items())
    length, count = [], []
    for k, v in stats:
        length.append(k)
        count.append(v['count'])
    plt.xlabel('word length')
    plt.ylabel('count')
    plt.xticks(range(1, 21, 1))
    plt.bar(length, count)
    plt.savefig('./count.jpg')


if __name__ == '__main__':
    file = '/mgData2/yangb/icefall/egs/arabic/ASR/zipformer/exp_lr_epoch_5.5_fp16/streaming/greedy_search/errs-mgb2_test-greedy_search-epoch-25-avg-15-chunk-16-left-context-128-use-averaged-model.txt'
    stats = word_stats(file)
    plot_stats(stats)
    plot_count(stats)
            
                
                