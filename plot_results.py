import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp


def get_loss_data(loss_df, size):
    order = ['none', 'xtra', 'full', 'half', 'wrong', 'mix']
    size_indices = loss_df.index[loss_df[0] == size]
    losses = np.zeros((int(len(size_indices)/11), 11))

    for count, si in enumerate(size_indices):
        si_order = order.index(loss_df[1][si])
        si_beacon = 1 if si_order else 0
        si_play = 0 if str(loss_df[2][si]) == 'nan' else 1
        losses[count//11, (2*si_order) - (1*si_beacon) + si_play] = np.sum(loss_df.iloc[si+1].to_numpy(dtype=float))

    return losses


data_path = "results/no_noise_image4/"

train_size = [5, 10, 20, 30, 40, 80]
methods0 = ['baseline', 'extra', 'exact (xyz)', 'distance', 'random', 'mixed']

results0 = {key: {'size': [], 'reward': []} for key in range(len(methods0))}
for ts in train_size:

    # train_file = data_path + "losses.csv"
    # train_df = pd.read_csv(train_file, header=None)
    # train_loss = get_loss_data(train_df, ts)

    test_file = data_path + "reward_" + str(ts) + "_dynamic.csv"
    test_df = pd.read_csv(test_file, header=None)
    test_df['size'] = ts

    data_cols = [0, 1, 3, 5, 7, 9]
    # data_cols = [0, 2, 4, 6, 8, 10]  # PLAY
    for dc_idx, dc in enumerate(data_cols):
        reward_data = test_df[dc].to_list()  # [train_loss[:, dc] < 1].to_list()
        results0[dc_idx]['reward'] += reward_data
        results0[dc_idx]['size'] += np.repeat(ts, len(reward_data)).tolist()


data_path = "results/no_noise_image5/"

train_size = [10, 30]
methods1 = ['baseline', 'extra', 'exact (xyz)', 'partial', 'random', 'mixed', 'other', 'distance']

results1 = {key: {'size': [], 'reward': []} for key in range(len(methods1))}
for ts in train_size:

    # train_file = data_path + "losses.csv"
    # train_df = pd.read_csv(train_file, header=None)
    # train_loss = get_loss_data(train_df, ts)

    test_file = data_path + "reward_" + str(ts) + "_dynamic.csv"
    test_df = pd.read_csv(test_file, header=None)
    test_df['size'] = ts

    data_cols = [0, 1, 3, 5, 7, 9, 11, 13]
    # data_cols = [0, 2, 4, 6, 8, 10, 12, 14]  # PLAY
    for dc_idx, dc in enumerate(data_cols):
        reward_data = test_df[dc].to_list()  # [train_loss[:, dc] < 1].to_list()
        results1[dc_idx]['reward'] += reward_data
        results1[dc_idx]['size'] += np.repeat(ts, len(reward_data)).tolist()


# plt.figure()
# for m in range(len(methods)):
#     df = pd.DataFrame(results[m])
#     # sns.lineplot(data=df, x=len(df.columns)-1, y=m, label=methods[m], linewidth=3)
#     sns.lineplot(data=df, x='size', y='reward', label=methods[m], linewidth=3)
# plt.legend(loc='lower right')
# plt.xticks(train_size)
# plt.xlabel("Train demos")
# plt.ylabel("Reward")
# # plt.ylim(12, 20)
# # plt.savefig('plots/results_nonoise_image_dynamic.png', dpi=1080)
# plt.show()

# bar plot
num_demos = 10

baseline = [r for ri, r in enumerate(results0[0]['reward']) if results0[0]['size'][ri] == num_demos]
extra = [r for ri, r in enumerate(results0[1]['reward']) if results0[1]['size'][ri] == num_demos]
exact = [r for ri, r in enumerate(results0[2]['reward']) if results0[2]['size'][ri] == num_demos]
half = [r for ri, r in enumerate(results1[3]['reward']) if results1[3]['size'][ri] == num_demos]
random = [r for ri, r in enumerate(results0[4]['reward']) if results0[4]['size'][ri] == num_demos]
mixed = [r for ri, r in enumerate(results0[5]['reward']) if results0[5]['size'][ri] == num_demos]
other = [r for ri, r in enumerate(results1[6]['reward']) if results1[6]['size'][ri] == num_demos]


X = ['baseline', 'extra', 'exact (xyz)', 'half', 'random', 'mixed', 'other']
Y = [np.mean(baseline), np.mean(extra), np.mean(exact), np.mean(half), np.mean(random), np.mean(mixed), np.mean(other)]
Y_err = [sp.stats.sem(baseline), sp.stats.sem(extra), sp.stats.sem(exact), sp.stats.sem(half),
         sp.stats.sem(random), sp.stats.sem(mixed), sp.stats.sem(other)]

plt.figure()
plt.bar(X, Y, yerr=Y_err)
plt.ylim(12, 18.5)
# plt.xticks([])
plt.savefig('plots/partial_results.png', dpi=1080)
plt.show()

print("Done.")
