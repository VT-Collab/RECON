import pickle
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def generate_scenario():

    # robot starts in center
    robot_x = [0., 0.]

    # place objects in a circle
    theta = 2*np.pi*np.random.random() - np.pi
    obj0 = [10.*np.cos(theta), 10.*np.sin(theta)]
    obj1 = [10.*np.cos(theta + np.pi/2), 10.*np.sin(theta + np.pi/2)]
    obj2 = [10.*np.cos(theta + np.pi), 10.*np.sin(theta + np.pi)]
    obj3 = [10.*np.cos(theta - np.pi/2), 10.*np.sin(theta - np.pi/2)]

    # choose goal based on color
    obj_colors = 5.*np.ones(4)
    attract_idx = np.random.choice(range(4))
    obj_colors[attract_idx] = 0.
    repel_idx = attract_idx - 2
    obj_colors[repel_idx] = 10.

    return np.array(robot_x + obj0 + obj1 + obj2 + obj3 + obj_colors.tolist())


def generate_demo(state, b_obj, dynamic=False):

    if dynamic:
        move_theta = np.pi/24.
    else:
        move_theta = 0.

    xi = []
    for _ in range(TIME_STEPS):

        # move robot away from object
        if np.linalg.norm(state[:2]) >= 10.:
            action = np.array([0., 0.])
        else:
            obj_state = state[(2*b_obj + 2):(2*b_obj + 4)]
            action = state[:2] - obj_state
            action /= np.linalg.norm(action)

        xi.append(np.concatenate((state, action), axis=None))

        # update robot state
        state[:2] += action

        # move other objects along the circle
        for obj_idx in range(4):
            theta = np.arctan2(state[2*obj_idx + 3], state[2*obj_idx + 2])
            new_theta = theta + move_theta
            state[2*obj_idx + 2] = 10*np.cos(new_theta)
            state[2*obj_idx + 3] = 10*np.sin(new_theta)

    return np.array(xi)


def create_img(state, image):

    img = deepcopy(image)

    state2img = 32. * (state[:10] + 12.) / 24.

    for idx in range(5):
        x_idx = round(state2img[2*idx])
        y_idx = round(state2img[2*idx + 1])

        for x in range(x_idx-1, x_idx+1, 1):
            for y in range(y_idx-1, y_idx+1, 1):

                x = np.clip(x, 0, 31)
                y = np.clip(y, 0, 31)

                if idx == 0:
                    img[0, x, y] = 0
                    img[1, x, y] = 0

                else:
                    red = (state[10+idx-1]/10.) * 255
                    green = (1. - state[10+idx-1]/10.) * 255
                    img[0, x, y] = red
                    img[1, x, y] = green
                    img[2, x, y] = 0

    return img


def plot_state(state):
    plt.plot(state[0], state[1], 'r*')
    plt.plot(state[2], state[3], 'o', color=str(state[10]/10.))
    plt.plot(state[4], state[5], 'o', color=str(state[11]/10.))
    plt.plot(state[6], state[7], 'o', color=str(state[12]/10.))
    plt.plot(state[8], state[9], 'o', color=str(state[13]/10.))
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])


TIME_STEPS = 10


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', type=int, default=80)
    args = parser.parse_args()

    half_choice = np.random.choice([0, 1])

    # GENERATE DEMONSTRATIONS
    images, demos = [], []
    beacon_xtra, beacon_full, beacon_half, beacon_dist, beacon_other, beacon_wrong, beacon_mix = [], [], [], [], [], [], []
    for n_scenario in range(args.scenarios):

        # sample start state and objects
        scenario = generate_scenario()

        # image
        background = np.round(225 + 30 * np.random.random((3, 32, 32))).astype(np.int64)
        image = create_img(scenario, background)

        # plt.figure()
        # plt.imshow(np.moveaxis(image, [0, 1, 2], [2, 0, 1]))
        # # plt.savefig('plots/state.png', dpi=1080)
        # plt.show()

        n_demo = 0
        max_demos = 1
        while n_demo < max_demos:

            # change start state
            start_state = deepcopy(scenario)

            # user objective -- go away from white/red object
            repel_obj = np.argmax(start_state[10:])
            other_objs = list(range(4))
            other_objs.remove(repel_obj)
            other_obj = repel_obj - 1
            random_obj = np.random.choice(other_objs)

            # demonstration
            xi_star = generate_demo(start_state, repel_obj, dynamic=True)

            # beacon information -- (x,y) position
            repel_pos = xi_star[:, (2*repel_obj + 2):(2*repel_obj + 4)]
            other_pos = xi_star[:, (2 * other_obj + 2):(2 * other_obj + 4)]
            random_pos = xi_star[:, (2*random_obj + 2):(2*random_obj + 4)]

            b_xtra = np.concatenate((repel_pos, other_pos), axis=1)
            b_full = deepcopy(repel_pos)
            b_half = deepcopy(repel_pos[:, half_choice][:, np.newaxis])
            b_dist = np.linalg.norm(xi_star[:, :2] - repel_pos, axis=1)[:, np.newaxis]
            b_other = deepcopy(other_pos)
            b_wrong = deepcopy(random_pos)
            b_mix = np.concatenate((repel_pos, random_pos), axis=1)

            # save
            demos.append(deepcopy(xi_star))
            n_demo += 1

            image_data = [create_img(s[:14], background) for s in xi_star]
            images.append(image_data)

            beacon_xtra.append(b_xtra)
            beacon_full.append(b_full)
            beacon_half.append(b_half)
            beacon_dist.append(b_dist)
            beacon_other.append(b_other)
            beacon_wrong.append(b_wrong)
            beacon_mix.append(b_mix)

    # GENERATE PLAY DATA
    play_images, play_data = [], []
    play_xtra, play_full, play_half, play_wrong, play_mix = [], [], [], [], []
    play_other, play_dist = [], []
    for n_scenario in range(args.scenarios*3):

        # sample start state and objects
        scenario = generate_scenario()

        # image
        background = np.round(225 + 30 * np.random.random((3, 32, 32))).astype(np.int64)
        image = create_img(scenario, background)

        start_state = deepcopy(scenario)
        theta = 2*np.pi*np.random.random() - np.pi
        start_state[0] += 10 * np.random.random() * np.cos(theta)
        start_state[1] += 10 * np.random.random() * np.sin(theta)

        # user objective -- go away from white/red object
        repel_obj = np.argmax(start_state[10:])
        other_objs = list(range(4))
        other_objs.remove(repel_obj)
        other_obj = repel_obj - 1
        random_obj = np.random.choice(other_objs)

        # beacon information -- (x,y) position
        repel_pos = start_state[(2 * repel_obj + 2):(2 * repel_obj + 4)]
        other_pos = start_state[(2 * other_obj + 2):(2 * other_obj + 4)]
        random_pos = start_state[(2 * random_obj + 2):(2 * random_obj + 4)]

        b_xtra = np.concatenate((repel_pos, other_pos))
        b_full = deepcopy(repel_pos)
        b_half = [repel_pos[half_choice]]
        b_dist = [np.linalg.norm(start_state[:2] - repel_pos)]
        b_other = deepcopy(other_pos)
        b_wrong = deepcopy(random_pos)
        b_mix = np.concatenate((repel_pos, random_pos))

        play_data.append(start_state)
        play_images.append(image)
        play_xtra.append(b_xtra)
        play_full.append(b_full)
        play_dist.append(b_dist)
        play_half.append(b_half)
        play_other.append(b_other)
        play_wrong.append(b_wrong)
        play_mix.append(b_mix)

    # save data
    enumerate(['Baseline', 'Exact', 'Partial', 'Other', 'Wrong'])
    pickle.dump(demos, open("data/demos.pkl", "wb"))
    pickle.dump(images, open("data/images.pkl", "wb"))
    pickle.dump(beacon_xtra, open("data/beacon_Exact_Other.pkl", "wb"))
    pickle.dump(beacon_full, open("data/beacon_Exact.pkl", "wb"))
    pickle.dump(beacon_half, open("data/beacon_Partial.pkl", "wb"))
    pickle.dump(beacon_dist, open("data/beacon_Dist.pkl", "wb"))
    pickle.dump(beacon_other, open("data/beacon_Other.pkl", "wb"))
    pickle.dump(beacon_wrong, open("data/beacon_Random.pkl", "wb"))
    pickle.dump(beacon_mix, open("data/beacon_Exact_Random.pkl", "wb"))

    pickle.dump(np.array(play_data), open("data/play_data.pkl", "wb"))
    pickle.dump(np.array(play_images), open("data/play_images.pkl", "wb"))
    pickle.dump(np.array(play_xtra), open("data/play_Exact_Other.pkl", "wb"))
    pickle.dump(np.array(play_full), open("data/play_Exact.pkl", "wb"))
    pickle.dump(np.array(play_half), open("data/play_Partial.pkl", "wb"))
    pickle.dump(np.array(play_dist), open("data/play_Dist.pkl", "wb"))
    pickle.dump(np.array(play_other), open("data/play_Other.pkl", "wb"))
    pickle.dump(np.array(play_wrong), open("data/play_Random.pkl", "wb"))
    pickle.dump(np.array(play_mix), open("data/play_Exact_Random.pkl", "wb"))

    print("Generated", len(demos), "demonstrations.")
