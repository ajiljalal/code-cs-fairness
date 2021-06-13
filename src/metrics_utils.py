"""Some utils for plotting metrics"""
# pylint: disable = C0111


import glob
import numpy as np
import utils
import matplotlib.pyplot as plt
# import seaborn as sns


def int_or_float(val):
    try:
        return int(val)
    except ValueError:
        return float(val)


def get_figsize(is_save):
    if is_save:
        figsize = [6, 4]
    else:
        figsize = None
    return figsize

def get_data(expt_dir):
    data = {}
    measurement_losses = utils.load_if_pickled(expt_dir + '/measurement_losses.pkl')
    l2_losses = utils.load_if_pickled(expt_dir + '/l2_losses.pkl')
    lpips_scores = utils.load_if_pickled(expt_dir + '/lpips_scores.pkl')
    likelihoods = utils.load_if_pickled(expt_dir + '/likelihoods.pkl')
    z_norms = utils.load_if_pickled(expt_dir + '/z_norms.pkl')
    data = {'measurement': measurement_losses.values(),
            'l2': l2_losses.values(),
            'lpips': lpips_scores.values(),
            'likelihood': likelihoods.values(),
            'norm': z_norms.values()}
    return data


def get_metrics(expt_dir):
    data = get_data(expt_dir)

    metrics = {}

    measurement_list = np.array(list(data['measurement']))
    measurement_list = measurement_list/(3*256*256)
    m_loss_mean = np.mean(measurement_list)
    m_loss_std = np.std(measurement_list) / np.sqrt(len(data['measurement']))
    metrics['measurement'] = {'mean': m_loss_mean, 'std': m_loss_std}

    l2_list = list(data['l2'])
    l2_loss_mean = np.mean(l2_list)
    l2_loss_std = np.std(np.array(l2_list)) / np.sqrt(len(data['l2']))
    metrics['l2'] = {'mean':l2_loss_mean, 'std':l2_loss_std}



    lpips_list = list([i for i in data['lpips']])
    v_loss_mean = np.mean(lpips_list)
    v_loss_std = np.std(lpips_list) / np.sqrt(len(data['lpips']))
    metrics['lpips'] = {'mean': v_loss_mean, 'std': v_loss_std}

    likelihood_list = list(data['likelihood'])
    likelihood_loss_mean = np.mean(likelihood_list)
    likelihood_loss_std = np.std(np.array(likelihood_list)) / np.sqrt(len(data['likelihood']))
    metrics['likelihood'] = {'mean':likelihood_loss_mean, 'std':likelihood_loss_std}

    norm_list = list(data['norm'])
    norm_loss_mean = np.mean(norm_list)
    norm_loss_std = np.std(np.array(norm_list)) / np.sqrt(len(data['norm']))
    metrics['norm'] = {'mean':norm_loss_mean, 'std':norm_loss_std}
    return metrics


def get_expt_metrics(expt_dirs):
    expt_metrics = {}
    for expt_dir in expt_dirs:
        metrics = get_metrics(expt_dir)
        expt_metrics[expt_dir] = metrics
    return expt_metrics


def get_nested_value(dic, field):
    answer = dic
    for key in field:
        answer = answer[key]
    return answer


def find_best(pattern, criterion, retrieve_list):
    dirs = glob.glob(pattern)
    metrics = get_expt_metrics(dirs)
    best_merit = 1e10
    answer = [None]*len(retrieve_list)
    for dir, val in metrics.items():
        merit = get_nested_value(val, criterion)
        if merit < 0:
            merit = -1 * merit
        if merit < best_merit:
            best_merit = merit
            best_dir = dir
            for i, field in enumerate(retrieve_list):
                answer[i] = get_nested_value(val, field)

    try:
        print(best_dir)
    except:
        best_dir = None
        pass
    return answer, best_dir


def plot(base, regex, criterion, retrieve_list, label, color=None):
    # todo
    marker_list = {'b':'s', 'r':'o', 'y':'^'}
    keys = map(int_or_float, [a.split('/')[-1] for a in glob.glob(base + '*')])
    means, std_devs = {}, {}
    for i, key in enumerate(keys):
        pattern = base + str(key) + regex
        answer, _ = find_best(pattern, criterion, retrieve_list)
        if answer[0] is not None:
            means[key], std_devs[key] = answer
    plot_keys = sorted(means.keys())
    if retrieve_list[0][0] != 'measurement':
        means = np.asarray([  means[key] for key in plot_keys])
        std_devs = np.asarray([  std_devs[key] for key in plot_keys])
        if color is None:
            (lines, caps, _) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
                                    marker='o', markersize=5, capsize=5, label=label)
        else:
            (lines, caps, _) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
                                    marker=marker_list[color], markersize=5, capsize=5, color=color, label=label)

    elif retrieve_list[0][0] == 'measurement':
        means = np.asarray([means[key] for key in plot_keys])
        std_devs = np.asarray([std_devs[key] for key in plot_keys])
        if color is None:
            (lines, caps, _) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
                                    fmt=':^', markersize=5, capsize=5, label=label)
        else:
            (lines) = plt.plot(plot_keys, means,
                                    ':^', markersize=5, color=color, label=label)
            # (lines, caps, _) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
            #                         fmt=':^', markersize=5, capsize=5, color=color)

    try:
        for cap in caps:
            cap.set_markeredgewidth(1)
        return lines.get_color()
    except:
        pass
