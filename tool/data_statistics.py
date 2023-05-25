import matplotlib.pyplot as plt

train_labels_path = './data/DISFA/list/DISFA_train_label_fold1.txt'
test_labels_path = './data/DISFA/list/DISFA_test_label_fold1.txt'

with open(train_labels_path) as f:
    train_lines = [line for line in f]

with open(test_labels_path) as f:
    test_lines = [line for line in f]

au_dict = {
    0: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    1: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    2: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    3: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    4: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    5: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    6: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
    7: {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0},
}

label_dict = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0}

zero_frames = 0

for line in train_lines:
    line = line.split()
    zero = True
    for i, l in enumerate(line):
        label = int(l)
        if label != 0:
            zero = False
        au_dict[i][label] += 1
        label_dict[label] += 1
    if zero:
        zero_frames += 1

for line in test_lines:
    line = line.split()
    zero = True
    for i, l in enumerate(line):
        label = int(l)
        if label != 0:
            zero = False
        au_dict[i][label] += 1
        label_dict[label] += 1
    if zero:
        zero_frames += 1

print(label_dict)
print(f'zero frames: {zero_frames}')
print(f'total frames: {len(train_lines) + len(test_lines)}')

# ------------------------------------------------------------
# plot number of instances for each level of AU activation summed over all AUs
# ------------------------------------------------------------
# plt.bar(range(len(label_dict)), list(label_dict.values()), align='center')
# plt.xticks(range(len(label_dict)), list(label_dict.keys()))
# plt.xlabel('Activation Intensity')
# plt.ylabel('Instances')
# plt.show()

# ------------------------------------------------------------
# plot number of occurrences of AU intensity levels for each FAU
# ------------------------------------------------------------
intensity = 5
plt.bar(range(8), [au_dict[x][intensity] for x in au_dict.keys()])
plt.xticks(range(8), [1, 2, 4, 6, 9, 12, 25, 26])
plt.xlabel('Facial Action Unit')
plt.ylabel('Number of Occurrences')
plt.title(f'Occurrences of Intensity Level {intensity} for Each FAU')
plt.show()
