import numpy as np
import os
import pickle
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from scipy import io, special
from sklearn import metrics

# Globals

tstep = 0.025
tstart = 0.025
tmax = 1.0
tnum = ((tmax - tstart) / tstep) + 1
thresholds = np.linspace(tstart, 1.0, np.round(tnum).astype(int))


MS_MAPPING = {"Wake": 0, "MS": 1}
AASM_MAPPING = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}



# Helper functions

def nans(shape):
    out = np.empty(shape)[0]
    out[:] = np.nan
    return out

def get_probs(file):
    out = np.load(file)
    probs = special.softmax(out, 1)
    return probs


def aasm_to_wake_sleep(aasm: np.array):
    out = np.zeros(aasm.shape)
    out[aasm > 0] = 1
    return out


def get_hit_stats(y_hat, y_true):

    tn = (y_hat == 0) * (y_true == 0)
    tp = (y_hat == 1) * (y_true == 1)
    fn = (y_hat == 0) * (y_true == 1)
    fp = (y_hat == 1) * (y_true == 0)

    hits = np.zeros(y_hat.shape)
    hit_map = {"TN": 0, "TP": 1, "FP": 2, "FN": 3}

    hits[tn] = hit_map["TN"]
    hits[tp] = hit_map["TP"]
    hits[fp] = hit_map["FP"]
    hits[fn] = hit_map["FN"]

    return hits, hit_map


def resample_usleep_preds(y_pred: np.array, data_per_pred: int, org_fs: int = 200, usleep_fs: int = 128):
    return np.repeat(y_pred, np.floor(data_per_pred * (org_fs / usleep_fs)), 0)


def load_pickle_from_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


def write_to_pickle_file(obj, file):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _find_singles(y, idx):
    pidx = idx - 1
    if pidx[0] < 0:
        prev = y[pidx[1:]]
        prev = np.append(0, prev)
    else:
        prev = y[pidx]

    nidx = idx + 1
    if nidx[-1] >= len(y):
        nxt = y[nidx[:-1]]
        nxt = np.append(nxt, 0)
    else:
        nxt = y[nidx]

    single_idx = np.logical_not(prev) * np.logical_not(nxt)
    singles = idx[single_idx]
    return singles


def get_target_label_start_and_stop_indices(labels, target):
    # Find indices of the target in labels
    idx = np.where(labels == target)[0]

    # Calculate where there are islands of target in labels
    islands = np.diff(idx) == 1
    ffy = np.pad(islands, pad_width=(1, 1), mode="constant", constant_values=(0, 0)).astype(int)

    # Calculate where the islands start (>0) and end (<0) by calculating the differnece of the island indices
    diff_idx = np.diff(ffy)

    # Find where they start and end
    start_idx = np.where(diff_idx > 0)[0]
    stop_idx = np.where(diff_idx < 0)[0]

    start = idx[start_idx]
    stop = idx[stop_idx]

    # Also find single targets
    if np.any(idx):
        singles = _find_singles(labels, idx)
    else:
        singles = np.empty([0])

    return start, stop, singles


def remove_invalid_labels(labels, target=1, min_duration=3, max_duration=15, fs=1, verbose=True):
    start, stop, singles = get_target_label_start_and_stop_indices(labels, target)
    target_time = (stop - start) / fs

    too_short = np.where(target_time < min_duration)[0]
    too_long = np.where(target_time > max_duration)[0]

    unit = "samples" if fs == 1 else "seconds"

    if verbose:
        print(f"{len(too_short)} labels are shorter than {min_duration} {unit}\n"
              f"{len(too_long)} labels are longer than {max_duration} {unit}\n")

    invalid_idx = np.hstack([too_short, too_long])

    fixed_labels = labels
    for i in invalid_idx:
        fixed_labels[start[i]:stop[i] + 1] = 0

    if np.any(singles):
        if verbose:
            print(f"{len(singles)} singletons were found and removed")
        fixed_labels[singles] = 0

    return fixed_labels


# def _fill_label_gaps(labels, target, limit, fs = 1):
#    
#    start, stop = get_target_label_start_and_stop_indices(labels, target)
#    label_gap = (start[1:] - stop[0:-1]) / fs
#    idx = np.where(ms_gap <= limit)[0]

#    filled_gaps = np.copy(labels)
#    for i in idx:
#        filled_gaps[stop[i]+1:start[i+1]] = 1
#    
#    return filled_gaps

def rolling_window(array, window_size, freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], freq)]


def get_all_probs(rec):
    probs = get_probs(rec)
    probs_sum = np.column_stack([probs[:, 0], np.sum(probs[:, 1:5], axis=1)])
    probs_max = np.column_stack([probs[:, 0], np.max(probs[:, 1:5], axis=1)])

    return probs, probs_sum, probs_max


def psuedo_resample(y_org, first_last):
    if len(y_org.shape) > 1:
        return np.array([np.median(y_org[:, x[0]:x[1]], 1) for x in first_last]).T
    else:
        return np.array([np.median(y_org[x[0]:x[1]]) for x in first_last])

def plot_roc_curve(y_true, y_probs, pos_label=1, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs[:, pos_label], pos_label=pos_label)

    ax.plot([0, 1], [0, 1])
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")

    return fpr, tpr, ax


def plot_precision_recall_curve(y_true, y_probs, pos_label=1, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_probs[:, pos_label], pos_label=pos_label)

    if ax is not None:
        baseline = np.sum(y_true == pos_label) / len(y_true)
        ax.plot([0, 1], [baseline, baseline], linestyle='--')
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")

    return precision, recall, ax


def compute_performance_metrics(y_true, y_pred, y_probs,
                                labels=[0, 1], classes=["Wake", "MS"],
                                minority_label=1, pos_label=1, plot_on=True):
    if plot_on:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=[9, 4])
    else:
        ax1 = None
        ax2 = None

    # Get general classification report
    report = metrics.classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                           output_dict=True)

    # Calculate ROC metics    
    roc_fpr, roc_tpr, roc_ax = plot_roc_curve(y_true, y_probs, pos_label, ax1)
    roc_auc = metrics.roc_auc_score(y_true, y_probs[:, pos_label])

    # Calculate PR curve metrics
    _precision, _recall, pr_ax = plot_precision_recall_curve(y_true, y_probs, pos_label, ax2)
    pr_auc = metrics.auc(_recall, _precision)

    # Compute Matthews correlation coefficent
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    # Compute Cohen's Kappa
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)

    # Store results in report
    report["roc_auc"] = roc_auc
    report["pr_auc"] = pr_auc
    report["mcc"] = mcc
    report["cohen_kappa"] = cohen_kappa

    # A bit extra spicy
    if plot_on:
        ax1.annotate(f"ROC AUC = {roc_auc:.2f}", (0.7, 0.2), fontsize=8)
        ax2.annotate(f"PR AUC = {pr_auc:.2f}", (0.7, 0.7), fontsize=8)

    return report


def my_collector(collection, ids, key, rm=True, min_dur=1, max_dur=np.inf):
    # print(f"Removing invalid labels: {rm}")
    i = 0
    for k, sub_collection in collection.items():
        if k not in ids:
            # print(f"Skipping: {k}")
            continue

        v = sub_collection[key]
        _y = sub_collection["labels"]

        if rm:
            v = np.array(
                [remove_invalid_labels(vx, min_duration=min_dur, max_duration=max_dur, fs=5, verbose=False) for vx in
                 v])
            _yy = remove_invalid_labels(_y, min_duration=min_dur, max_duration=max_dur, fs=5, verbose=False)

        if i == 0:
            y_hat = v
            y = _y
        else:
            y_hat = np.column_stack([y_hat, v]) if len(y_hat.shape) > 1 else np.hstack([y_hat, v])
            y = np.hstack([y, _y])
        i += 1
    return y_hat, y


# Plotting functions

def format_ax(ax, labs):
    ax.set_xlabel("Period number")
    ax.set_ylabel("Sleep stage")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    line = ax.lines[0]
    ids = line.get_xdata()
    ax.set_xlim(1, ids[-1] + 1)
    l = ax.legend(loc=3)
    l.get_frame().set_linewidth(0)


def ghost_poly(face_color=[1, 1, 1], edge_color=[0, 0, 0]):
    _xy = np.empty([4, 2])
    _xy[:] = np.nan
    _poly = Polygon(_xy, facecolor=face_color, edgecolor=edge_color)
    return _poly


def plot_probs(ax, probs: np.array, labs, fs=1):
    av = np.cumsum(probs, axis=1)
    c = sns.color_palette("tab10", len(labs) - 1)

    # Create 'ghost' patch for 'Wake'
    _poly = ghost_poly()
    ax.add_patch(_poly)

    for i in range(probs.shape[1] - 1):
        xy = np.zeros([av.shape[0] * 2, 2])
        xy[:av.shape[0], 0] = np.arange(av.shape[0]) / fs
        xy[av.shape[0]:, 0] = np.flip(np.arange(av.shape[0]), axis=0) / fs
        xy[:av.shape[0], 1] = av[:, i]
        xy[av.shape[0]:, 1] = np.flip(av[:, i + 1], axis=0)

        poly = Polygon(xy, facecolor=c[i], edgecolor=None)
        ax.add_patch(poly)

    ax.set_ylabel("Probability")
    ax.legend(labs, loc='lower left')
    ax.set_xlim([0, av.shape[0]])

def MatplotlibClearMemory():
    allfignums = plt.get_fignums()
    for i in allfignums:
        fig = plt.figure(i)
        fig.clear()
        plt.close(fig)


# Plotting functions

def format_ax(ax, labs):
    ax.set_xlabel("Period number")
    ax.set_ylabel("Sleep stage")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    line = ax.lines[0]
    ids = line.get_xdata()
    ax.set_xlim(1, ids[-1] + 1)
    l = ax.legend(loc=3)
    l.get_frame().set_linewidth(0)


def ghost_poly(face_color=[1, 1, 1], edge_color=[0, 0, 0]):
    _xy = np.empty([4, 2])
    _xy[:] = np.nan
    _poly = Polygon(_xy, facecolor=face_color, edgecolor=edge_color)
    return _poly


def plot_probs(ax, probs: np.array, labs, fs=1):
    av = np.cumsum(probs, axis=1)
    c = sns.color_palette("tab10", len(labs) - 1)

    # Create 'ghost' patch for 'Wake'
    _poly = ghost_poly()
    ax.add_patch(_poly)

    for i in range(probs.shape[1] - 1):
        xy = np.zeros([av.shape[0] * 2, 2])
        xy[:av.shape[0], 0] = np.arange(av.shape[0]) / fs
        xy[av.shape[0]:, 0] = np.flip(np.arange(av.shape[0]), axis=0) / fs
        xy[:av.shape[0], 1] = av[:, i]
        xy[av.shape[0]:, 1] = np.flip(av[:, i + 1], axis=0)

        poly = Polygon(xy, facecolor=c[i], edgecolor=None)
        ax.add_patch(poly)

    ax.set_ylabel("Probability")
    ax.legend(labs, loc='lower left')
    ax.set_xlim([0, av.shape[0]])


def plot_label_patches(labels, mapping, y_min=0, y_max=1, ax=None, fs=1):
    if ax is None:
        _, ax = plt.subplots()

    #     def create_patches(ax, start, stop, cc, y_min=0, y_max=1, fs=1):
    #         for start_idx, stop_idx in zip(start, stop):
    #             x = np.arange(start_idx, stop_idx + 1) / fs
    #             xn = len(x)
    #             xy = np.zeros([xn * 2, 2])
    #             xy[:xn, 0] = x
    #             xy[xn:, 0] = np.flip(x, axis=0)

    #             xy[:xn, 1] = np.ones([xn]) * y_min
    #             xy[xn:, 1] = np.ones([xn]) * y_max
    #             poly = Polygon(xy, facecolor=cc, edgecolor=None, alpha=1)
    #             patch = ax.add_patch(poly)

    #         return patch

    def fill_between(ax, x, y, target, cc, y_min=0, y_max=1, fs=1):
        area = ax.fill_between(x, y_min, y_max,
                               where=y == target,
                               color=cc)
        return area

    handles = []
    handle_names = []
    colors = sns.color_palette("tab10", len(mapping.keys()))
    for i, (label_name, label) in enumerate(mapping.items()):
        # Faster method
        x = np.arange(0, len(labels)) / fs
        ax.step(x, labels == label, color=colors[label])
        handle = fill_between(ax, x, labels, label, colors[label], y_min, y_max, fs=fs)
        handles.append(handle)
        handle_names.append(label_name)

    #        _, _, singles = get_target_label_start_and_stop_indices(labels, label)
    #         # Convert to time axis with fs
    #         start = start
    #         stop = stop
    #         if not np.any(start):
    #             print("No start")
    #             continue
    #         else:

    #             handle = create_patches(ax, start, stop, colors[label], y_min, y_max, fs=fs)
    #             handles.append(handle)
    #             handle_names.append(label_name)

    #         if np.any(singles):
    #             for x in singles:
    #                 _ = ax.axvline(x=x / fs, ymin=y_min, ymax=y_max, color=colors[label])

    return handles, handle_names


def MatplotlibClearMemory():
    allfignums = plt.get_fignums()
    for i in allfignums:
        fig = plt.figure(i)
        fig.clear()
        plt.close(fig)

def set_to_skorucack_time(ax):
    new_x_ticks = np.arange(0,45,5)
    ax.set_xticks(new_x_ticks*60)
    ax.set_xticklabels(new_x_ticks)


class BernLabels(object):
    folder = "labels"
    raw_mapping = {"Wake": 0,
                   "MSE": 1,
                   "MSEc": 2,
                   "ED": 3}

    min_duration = 3
    max_duration = 15

    def __init__(self, file: str, mapping: dict, include_unilateral=False):

        # File settings and read
        self.file = file
        self.path = os.path.join(self.folder, file) if not os.path.exists(self.file) else file
        self.__raw = io.loadmat(self.path)

        # Disseminate raw data
        self.fs = 200
        self.raw_fs = 200
        self.raw_O1 = np.squeeze(self.__raw['labels']['O1'][0][0])
        self.raw_O2 = np.squeeze(self.__raw['labels']['O2'][0][0])
        self.raw_labels = np.vstack([self.raw_O1, self.raw_O2])
        self.num_labels = self.raw_labels.shape[1]
        self.time = np.arange(0, (self.num_labels) / self.fs, 1 / self.fs)

        # Apply mapping (i.e. convert   Bern to Wake vs Sleep)
        self.mapping = mapping
        self.include_unilateral = include_unilateral
        self.label_mapping = {k: i for i, k in enumerate(self.mapping.keys())}
        self.convert_labels(mapping, include_unilateral)

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(file='{self.file}', mapping={self.mapping}, include_bilateral={self.include_unilateral})"

    def append(self, other):

        assert self.mapping == other.mapping
        assert self.include_unilateral == other.include_unilateral

        self.file = [self.file, other.file]
        self.__raw = [self.__raw, other.__raw]
        func = lambda x, y: np.hstack([x, y])
        self.raw_O1 = func(self.raw_O1, other.raw_O2)
        self.raw_O2 = func(self.raw_O2, other.raw_O2)
        self.raw_labels = func(self.raw_labels, other.raw_labels)
        self.num_labels = self.num_labels + other.num_labels
        self.time = np.arange(0, (self.num_labels) / self.fs,
                              1 / self.fs)  # Incorrect but needed for plotting hypnograms

        self.labels = func(self.labels, other.labels)

        return self

    def convert_labels(self, mapping=dict, crit=dict):

        print_on = False

        self.labels = np.empty(self.raw_O1.shape)
        self.labels[:] = np.nan

        wake_crit = crit["Wake"]
        ms_crit = crit["MS"]

        wake_idx = wake_crit(np.isin(self.raw_labels, mapping["Wake"]))
        ms_idx = ms_crit(np.isin(self.raw_labels, mapping["MS"]))

        self.labels[wake_idx] = self.label_mapping["Wake"]
        self.labels[ms_idx] = self.label_mapping["MS"]

        self.mapping = mapping

        return

    def apply_time_critera(self, min_duration=3, max_duration=15, replace=True):

        ms_index = self.label_mapping["MS"]
        copy_labels = np.copy(self.labels)
        fixed_labels = remove_invalid_labels(copy_labels, target=ms_index,
                                             min_duration=min_duration, max_duration=max_duration,
                                             fs=self.fs)

        if replace:
            print("Overwriting labels!")
            self.labels = fixed_labels
            return
        else:
            return fixed_labels

    def apply_rolling_func(self, win=0.2, step=0.2, func=None, replace=False):
        y = np.copy(self.labels)
        win_samples = int(win * self.fs)
        step_samples = int(step * self.fs)
        arr = np.array(rolling_window(y, win_samples, step_samples))

        if func is None:
            y_func = np.median(arr, 1)
        else:
            y_func = func(arr)

        if replace:
            self.labels = y_func
            self.prev_fs = self.fs
            self.fs = 1 / step
        else:
            return y_func, rolling_window(self.time, win_samples, step_samples)

    def plot_raw_labels(self, ax=None, as_hypnogram=False, fs=200):

        if ax is None:
            fig, ax = plt.subplots(figsize=[9, 5])

        if as_hypnogram:
            ax.step(self.time, self.raw_O1, "-", linewidth=1, color="darkred", label="Raw [O1]")
            ax.step(self.time, self.raw_O2, "--", linewidth=1, color="darkgray", label="Raw [O2]")
            ax.legend()
            format_ax(ax, self.raw_mapping.keys())

        else:
            _, _ = plot_label_patches(labels=self.raw_O1, mapping=self.raw_mapping, fs=fs,
                                      y_min=0, y_max=0.5, ax=ax)
            handles, _ = plot_label_patches(labels=self.raw_O2, mapping=self.raw_mapping, fs=fs,
                                            y_min=0.5, y_max=1, ax=ax, )
            #             ax.autoscale(enable=True, axis = "both", tight = True)
            ax.legend(handles, self.raw_mapping.keys())
            ax.set_yticks([0.25, 0.75])
            ax.set_yticklabels(["Raw O1", "Raw O2"], rotation=45)
            ax.set_ylim([0, 1])

        xlab = "Time [s]" if fs == self.raw_fs else "Sample #"
        ax.set_xlabel(xlab)
        ax.autoscale(tight=True)
        ax.plot([*ax.get_xlim()], [0.5, 0.5], 'k-', linewidth=0.5)

        return ax

    def plot_labels(self, ax=None, as_hypnogram=False, fs=200):

        if ax is None:
            fig, ax = plt.subplots(figsize=[9, 5])

        if as_hypnogram:
            ax.step(self.time, self.labels, "-", linewidth=1, color="darkblue", label="New Labels")
            ax.legend()
            format_ax(ax, self.mapping.keys())
            ax.set_xlabel("Time [s]")

        else:
            hdl, names = plot_label_patches(labels=self.labels, mapping=self.label_mapping, ax=ax, fs=fs)
            ax.legend(hdl, names, loc="upper left")
            ax.set_yticks([0.5])
            ax.set_yticklabels(["New labels"], rotation=45)

        xlab = "Time [s]" if fs == self.fs else "Sample #"
        ax.set_xlabel(xlab)
        ax.autoscale(tight=True)
        return ax

    # def __proxy_legend(self, ax):
    #     _colors = sns.color_palette("tab10", len(self.raw_mapping.keys()) - 1)
    #     _handles = [ax.add_patch(ghost_poly())]
    #     for i in range(len(self.raw_mapping.keys()) - 1):
    #         _poly = ghost_poly(face_color=_colors[i], edge_color=[1, 1, 1])
    #         _patch = ax.add_patch(_poly)
    #         _handles.append(_patch)
    #     return _handles

    def compare_conversion_to_raw_labels(self, ax1=None, ax2=None):
        if ax1 is None and ax2 is None:
            _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        self.plot_raw_labels(ax=ax1, fs=200)
        self.plot_labels(ax=ax2, fs=self.fs)
        ax1.autoscale(enable=True, axis="both", tight=True)
        ax.autoscale(enable=True, axis="both", tight=True)
        return ax1, ax2


# Trackers

class Tracker(object):

    def __init__(self, k, train_y_true, train_y_hat, val_y_true, val_y_hat):

        self.k = k

        tuning = len(train_y_hat.shape) > 1

        # Calculate training metrics
        loop_func = lambda yt, yh, f: [f(yt, y) for y in yh]
        static_func = lambda yt, yh, f: f(yt, yh)
        func = loop_func if tuning else static_func

        self._train_precision = func(train_y_true, train_y_hat, metrics.precision_score)
        self._train_recall = func(train_y_true, train_y_hat, metrics.recall_score)
        self._train_f1 = func(train_y_true, train_y_hat, metrics.f1_score)
        self._train_kappa = func(train_y_true, train_y_hat, metrics.cohen_kappa_score)

        if tuning:
            opt_idx = np.argmax(self._train_kappa) if len(train_y_hat.shape) > 1 else 0
            self.opt_idx = opt_idx
            self.opt_threshold = thresholds[opt_idx]

            self.train_opt_kappa = self._train_kappa[opt_idx]
            self.train_opt_f1 = self._train_f1[opt_idx]
            self.train_opt_recall = self._train_recall[opt_idx]
            self.train_opt_precision = self._train_precision[opt_idx]

        else:
            self.opt_idx = np.nan
            self.opt_threshold = np.nan

            self.train_opt_kappa = self._train_kappa
            self.train_opt_f1 = self._train_f1
            self.train_opt_recall = self._train_recall
            self.train_opt_precision = self._train_precision

        # Calculate validation metrics
        val_y_opt = val_y_hat[opt_idx, :] if tuning else val_y_hat

        self.val_precision = metrics.precision_score(val_y_true, val_y_opt)
        self.val_recall = metrics.recall_score(val_y_true, val_y_opt)
        self.val_f1 = metrics.f1_score(val_y_true, val_y_opt)
        self.val_kappa = metrics.cohen_kappa_score(val_y_true, val_y_opt)

    def to_dict(self):
        d = vars(self)
        out = {}
        for k, v in d.items():
            if not k.startswith("_"):
                out[k] = v
        return out


class EvaluationTracker(object):

    def __init__(self, y_true, y_hat, opt_idx=None):

        if np.any(opt_idx) and len(y_hat.shape) > 1:
            y_hat = y_hat[opt_idx, :]

        try:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_hat).ravel()
            self.specificity = tn / (tn + fp)
        except:
            pass

        self.accurcy = metrics.accuracy_score(y_true, y_hat)
        self.precision = metrics.precision_score(y_true, y_hat)
        self.recall = metrics.recall_score(y_true, y_hat)
        self.f1 = metrics.f1_score(y_true, y_hat)
        self.kappa = metrics.cohen_kappa_score(y_true, y_hat)
