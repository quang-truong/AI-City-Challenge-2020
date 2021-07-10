import pickle
import os

dataset_dir = 'vehiclereid/datasets/AIC20_ReID/image_train_cropped/'

f = open('vehiclereid/datasets/AIC20_ReID/train_track_id.txt')
lines = f.readlines()
track_list = {}
pid2track = {}
j = 0
big_imgs = []
f.close()

for line in lines:
    line = line[:-2]
    ls = line.split(" ")
    for i in range(len(ls)):
        print(ls[i])
        ls[i] += ".jpg"
        print(ls[i])
        ls[i] = ls[i].zfill(10)
        print(ls[i])
        ls[i] = (ls[i], os.path.getsize(dataset_dir + ls[i]))
    ls.sort(key= lambda filename: filename[1], reverse= True)
    for i in range(len(ls)):
        ls[i] = ls[i][0]
        pid2track[ls[i]] = j
        if (i == 0):
            big_imgs.append(ls[i])
    track_list[j] = ls
    j += 1

print(big_imgs)
print(len(big_imgs))

with open("train_track2pid.pkl", 'wb') as f:
    pickle.dump(track_list, f)

with open("train_pid2track.pkl", 'wb') as f:
    pickle.dump(pid2track, f)

with open("train_images.pkl", 'wb') as f:
    pickle.dump(big_imgs, f)
