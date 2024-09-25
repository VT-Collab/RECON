import torch
import pickle
import numpy as np
from model import BeaconVision
from get_data import generate_scenario, create_img, plot_state
from csv import writer
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse


def rollout_policy(my_model, start, dynamic=True, image=None):
    xi_hat = []
    state = deepcopy(start)
    for _ in range(TIME_STEPS):
        xi_hat.append(deepcopy(state))
        if image is None:
            pi = my_model.policy(torch.FloatTensor([state[:2]]),
                                 my_model.feature_encoder(torch.FloatTensor([state])))
        else:
            state_image = create_img(state, image)
            pi = my_model.policy(torch.FloatTensor([state[:2]]),
                                 my_model.feature_encoder(torch.FloatTensor([state_image])))
        state[:2] += pi.detach().numpy()[0]

        if dynamic:
            # move other objects along the circle
            for obj_idx in range(4):
                theta = np.arctan2(state[2 * obj_idx + 3], state[2 * obj_idx + 2])
                new_theta = theta + (np.pi/24)
                state[2 * obj_idx + 2] = 10 * np.cos(new_theta)
                state[2 * obj_idx + 3] = 10 * np.sin(new_theta)

    xi_hat = np.array(xi_hat)

    return xi_hat


def generate_img_traj(trajectory, background):

    img_traj = []
    for state in trajectory:
        img = create_img(state, background)
        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
        img_traj.append(img)

    return img_traj

def visualize_rollout(rollouts):
    
    fig, axs = plt.subplots(1, 5, figsize=(16, 5))
    fig.tight_layout()

    for i, b_type in enumerate(MODEL_LIST):
        axs[i].set_title(b_type)

    plt.ion()
    plt.show()

    traj_len = len(rollouts['Baseline'])

    for i in range(traj_len):
        axs[0].imshow(rollouts['Baseline'][i], interpolation='nearest')
        axs[1].imshow(rollouts['Exact'][i], interpolation='nearest')
        axs[2].imshow(rollouts['Partial'][i], interpolation='nearest')
        axs[3].imshow(rollouts['Other'][i], interpolation='nearest')
        axs[4].imshow(rollouts['Random'][i], interpolation='nearest')
        
        plt.draw()
        plt.pause(1)

    plt.close()


TIME_STEPS = 10
MODEL_LIST = ['Baseline', 'Exact', 'Partial', 'Other', 'Random']
# Args
parser = argparse.ArgumentParser()
parser.add_argument('--scenarios', type=int, default=100)
parser.add_argument('--render_freq', type=int, default=10)
args = parser.parse_args()
render_freq = args.render_freq

render = True
if render_freq <= 0:
    render = False
    render_freq = 1

# dataset
dataset = pickle.load(open("data/demos.pkl", "rb"))

# load models
# torch.manual_seed(0)
model = BeaconVision(1)
model.load_state_dict(torch.load('data/model.pt'))
model.eval()


# test rollout (final reward)
reward_result_dynamic = []

for i in range(args.scenarios):


    # Store rollouts
    rollouts = {}

    # generate random scenario
    scenario = generate_scenario()
    start_state = deepcopy(scenario)
    repel_obj = np.argmax(start_state[10:])

    background = np.round(225 + 30 * np.random.random((3, 32, 32))).astype(np.int64)

    # test with dynamic objects ---------------------------------------------
    xi_model = rollout_policy(model, start_state, dynamic=True, image=background)
    final_obj_state = xi_model[-2, (2*repel_obj + 2):(2*repel_obj + 4)]
    reward_model = np.linalg.norm(final_obj_state - xi_model[-1, :2])
    rollouts['Baseline'] = generate_img_traj(xi_model, background)
    # print('Baseline', reward_model)

    rr_dynamic = [reward_model]
    b_dims = [2, 1, 2, 2]


    for b_idx, b_type in enumerate(MODEL_LIST[1:]):
        beacon_model = BeaconVision(b_dims[b_idx])
        beacon_model.load_state_dict(torch.load('data/beacon_model_' + b_type + '.pt'))
        beacon_model.eval()

        xi_beacon = rollout_policy(beacon_model, start_state, dynamic=True, image=background)
        reward_beacon = np.linalg.norm(final_obj_state - xi_beacon[-1, :2])

        rollouts[b_type] = generate_img_traj(xi_beacon, background)

        rr_dynamic.append(reward_beacon)
        # print(b_type, reward_beacon)

    reward_result_dynamic.append(rr_dynamic)

    if render and (i % args.render_freq == 0):
        visualize_rollout(rollouts)


reward_result_dynamic = np.mean(reward_result_dynamic, axis=0)
print('Avg run results: [Baseline, Exact, Partial, Other, Random]')
print(np.round(reward_result_dynamic,3))
# save result
with open('results/reward_'+str(len(dataset))+'_dynamic.csv', 'a') as f:
    writer(f).writerow(reward_result_dynamic)
    f.close()
