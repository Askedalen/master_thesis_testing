import _pickle as pickle
import sys

train_filenames = pickle.load(open('train_filenames.pickle', 'rb'))
val_filenames = pickle.load(open('val_filenames.pickle', 'rb'))
test_filenames = pickle.load(open('test_filenames.pickle', 'rb'))

for check_file in sys.argv:
    if check_file == ".\chord_generator_test\control_filenames.py":
        continue
    print(f'Checking file {check_file}')

    if check_file in train_filenames:
        print('Found in train_filenames')
    if check_file in val_filenames:
        print('Found in val_filenames')
    if check_file in test_filenames:
        print('Found in test_filenames')
    print()