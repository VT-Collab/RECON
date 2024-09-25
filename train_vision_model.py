import torch
from torch.utils.data import Dataset, DataLoader
from model import BeaconVision
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from csv import writer


class MyData(Dataset):

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


parser = argparse.ArgumentParser()
parser.add_argument('--beacon', type=str, default="none")
parser.add_argument('--use_play', action=argparse.BooleanOptionalAction)
args = parser.parse_args()


# dataset
dataset = pickle.load(open("data/demos.pkl", "rb"))
train_demos = np.reshape(dataset, (len(dataset)*10, 16))
imageset = pickle.load(open("data/images.pkl", "rb"))
train_images = np.reshape(imageset, (len(imageset)*10, 3, 32, 32))

# beacon info
print(args.beacon)
if args.beacon == "none":
    use_beacon = False
    beacon_dims = 1
else:
    use_beacon = True
    infoset = pickle.load(open("data/beacon_" + args.beacon + ".pkl", "rb"))
    _, _, beacon_dims = np.shape(infoset)
    train_beacon = np.reshape(infoset, (len(infoset)*10, beacon_dims))
    play_beacon = pickle.load(open("data/play_" + args.beacon + ".pkl", "rb"))


# play data
train_play_data = pickle.load(open("data/play_data.pkl", "rb"))
train_play_images = pickle.load(open("data/play_images.pkl", "rb"))

# training parameters
l1_threshold = 0.1
l2_threshold = 1.0
if args.use_play:
    EPOCH = 6000
    LR = 4e-4
elif use_beacon:
    EPOCH = 6000
    LR = 4e-4
else:
    EPOCH = 3000
    LR = 4e-4
BATCH_SIZE = int(len(dataset)*10/5)

num_tries = 0
l1, l2 = np.inf, np.inf
while (l1 > l1_threshold or l2 > l2_threshold)  and num_tries < 5:

    num_tries += 1
    print("Trying", num_tries)

    # torch.manual_seed(0)

    # model and optimizer
    model = BeaconVision(beacon_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_data = MyData(train_demos)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # main training loop
    losses = []
    for epoch in range(EPOCH + 1):

        # get batch
        batch = np.random.choice(range(len(dataset)*10), BATCH_SIZE)

        x = torch.FloatTensor(train_demos[batch])
        images = torch.FloatTensor(train_images[batch])
        actions = x[:, 14:16]

        phi = model.feature_encoder(images)
        loss_1 = model.mse_func(model.policy(x[:, :2], phi), actions)

        if use_beacon or args.use_play:
            b_hat = model.predictor(phi)
            b_info = torch.FloatTensor(train_beacon[batch])
            loss_2 = model.mse_func(b_hat, b_info)

            if args.use_play:
                play_batch = np.random.choice(range(len(train_play_data)), BATCH_SIZE)
                play_images = torch.FloatTensor(train_play_images)[play_batch]
                play_b_info = torch.FloatTensor(play_beacon)[play_batch]
                phi = model.feature_encoder(play_images)

                b_hat = model.predictor(phi)
                loss_3 = model.mse_func(b_hat, play_b_info)
            else:
                loss_3 = torch.tensor(0.)

        else:
            loss_2, loss_3 = torch.tensor(0.), torch.tensor(0.)

        loss = (1.0*loss_1) + (1.2*loss_2) + (1.2*loss_3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 500 == 0:
            print(epoch, loss_1.item(), loss_2.item(), loss_3.item())

    l1 = loss_1.item()
    l2 = loss_2.item() + loss_3.item()

# plot training loss
# plt.figure()
# plt.plot(losses)
# plt.show()

with open('results/losses.csv', 'a') as f:
    writer(f).writerow([len(dataset), args.beacon, args.use_play])
    writer(f).writerow([loss_1.item(), loss_2.item(), loss_3.item()])
    f.close()

if use_beacon or args.use_play:
    if args.use_play:
        torch.save(model.state_dict(), "data/play_model_" + args.beacon + ".pt")
    else:
        torch.save(model.state_dict(), "data/beacon_model_" + args.beacon + ".pt")
else:
    torch.save(model.state_dict(), "data/model.pt")
