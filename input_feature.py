import os
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
from random import shuffle
import speechpy
import datetime
import h5py


######################################
####### Define the dataset class #####
######################################
class AudioDataset():
    """Audio dataset."""

    def __init__(self, filepath_list, list_label, sequence_size = 80):
        """
        Args:
            files_path (string): Path to the .txt file which the address of files are saved in it.
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.labels = list_label
        self.sequence_size = sequence_size

        # Open the .txt file and create a list from each line.
        # with open(files_path, 'r') as f:
        #     content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        list_files = []
        for x in filepath_list:
            sound_file_path = x
            try:
                with open(sound_file_path, 'rb') as f:
                    riff_size, _ = wav._read_riff_chunk(f)
                    file_size = os.path.getsize(sound_file_path)

                # Assertion error.
                assert riff_size == file_size and os.path.getsize(sound_file_path) > 1000, "Bad file!"

                # Add to list if file is OK!
                list_files.append(x.strip())
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print('file %s is corrupted!' % sound_file_path)
            # except:
            #     print("Unexpected error:", sys.exc_info()[0])
            #     raise

        # Save the correct and healthy sound files to a list.
        self.sound_files = list_files

    def __len__(self):
        return len(self.sound_files)

    def getitem(self, idx):
        # Get the sound file path
        sound_file_path = self.sound_files[idx]

        ##############################
        ### Reading and processing ###
        ##############################

        # Reading .wav file
        fs, signal = wav.read(sound_file_path)

        # Reading .wav file
        import soundfile as sf
        signal, fs = sf.read(sound_file_path)

        ###########################
        ### Feature Extraction ####
        ###########################

        # DEFAULTS:
        num_coefficient = 40

        # Staching frames
        frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025,
                                                  frame_stride=0.01,
                                                  zero_padding=True);
        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]
        logenergy = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
                                          num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                          high_frequency=None, num_cepstral = 40)
        ########################
        ### Handling sample ####
        ########################

        # Label extraction
        labels = self.labels

        print("logenergy")
        print(logenergy)
        print(len(logenergy))
        print(logenergy.shape)

        index = 0
        list_sequence = []
        list_label = []
        # print("\n --------------- logenergy[index:index+self.sequence_size*num_coefficient]")
        # print(logenergy[index:index+self.sequence_size*num_coefficient])
        # print(len(logenergy[index:index+self.sequence_size*num_coefficient]))
        # print(len(logenergy))
        print('***************')
        print(len(logenergy[0]))
        while index + self.sequence_size < logenergy.shape[0]:
            vector = logenergy[index:index+self.sequence_size,:]
            print("lalal {0}".format(len(vector)))
            print("lololo {0}".format(len(vector.reshape((1,self.sequence_size,num_coefficient,1)))))
            list_sequence.append(vector.reshape((1,self.sequence_size,num_coefficient,1)))
            list_label.append(np.array((labels[idx]), dtype=np.int32))
            index+= self.sequence_size 
        return list_sequence, list_label



def ExtractDataset(dev_folder_path, test_folder_path):
    # Process dev dataset
    data_user_tuple = ExtractUser(dev_folder_path)
    filepath_list = []
    list_label = []
    user_dictionnary = {}
    for d in data_user_tuple:
        current_label = d[0]
        current_foldername = d[1]
        current_folderpath = d[2]
        user_dictionnary[current_foldername] = current_label;
        filepath_list_local,list_label_local  = ExtractWav(current_folderpath, current_label)
        filepath_list =  filepath_list + filepath_list_local
        list_label =  list_label + list_label_local

    dataset = AudioDataset(filepath_list=filepath_list, list_label=list_label)
   
    utterrance_dev = []
    labels_dev = []

    for idx in range(len(list_label)):
        feature_, labels_ = dataset.getitem(idx)
        utterrance_dev += feature_
        labels_dev += labels_


    print('utterrance_dev ')
    print(len(utterrance_dev))   
    print('labels_dev ')
    print(len(labels_dev))

    utterrance_dev = np.vstack(utterrance_dev)
    labels_dev = np.vstack(labels_dev)


       # Process test dataset
    data_user_tuple = ExtractUser(test_folder_path)
    filepath_list = []
    list_label = []
    for d in data_user_tuple:
        current_label = d[0]
        current_foldername = d[1]
        current_folderpath = d[2]
        filepath_list_local,list_label_local  = ExtractWav(current_folderpath, current_label)
        filepath_list =  filepath_list + filepath_list_local
        list_label =  list_label + list_label_local

    dataset = AudioDataset(filepath_list=filepath_list, list_label=list_label)
   
    utterrance_test = []
    labels_test = []

    for idx in range(len(list_label)):
        feature_, labels_ = dataset.getitem(idx)
        utterrance_test += feature_
        labels_test += labels_

    utterrance_test = np.vstack(utterrance_test)
    labels_test = np.vstack(labels_test)

    return user_dictionnary, utterrance_dev, labels_dev, utterrance_test, labels_test


def ExtractUser(directory_path ):

    data_user_tuple = []
    index_label = 0
    for folder_name in os.listdir(directory_path):
        folder_path =  os.path.join(directory_path, folder_name)
        if os.path.isdir(folder_path): # check whether the current object is a folder or not
            data_user_tuple.append((index_label, folder_name, folder_path))
            index_label = index_label + 1

    print(data_user_tuple)
    return data_user_tuple


def ExtractWav(directory_path , label ):
    list_file = []
    list_label = []
    for filename in os.listdir(directory_path):
        folder_path =  os.path.join(directory_path, filename)
        if os.path.isdir(folder_path): # check whether the current object is a folder or not
            for wavfile in os.listdir(folder_path):
                wavfile_path = os.path.join(folder_path, wavfile)
                if os.path.isfile(wavfile_path) and os.path.splitext(wavfile)[1] == '.wav':
                    list_file.append(wavfile_path)
                    list_label.append(label)
    return list_file, list_label


if __name__ == '__main__':
    # add parser
    parser = argparse.ArgumentParser(description='Input pipeline')

    # The text file in which the paths to the audio files are available.
    # The path are relative to the directory of the audio files
    # Format of each line of the txt file is "class_label subject_dir/sound_file_name.ext"
    # Example of each line: 0 subject/sound.wav
    parser.add_argument('--dev_folder_path',
                        default=os.path.expanduser(
                            '/home/xavier/Desktop/developpement/Network/dataset/CelebVox/dev2/'),
                        help='The folder names for development phase')


    parser.add_argument('--test_folder_path',
                        default=os.path.expanduser(
                            '/home/xavier/Desktop/developpement/Network/dataset/CelebVox/test2/'),
                        help='The folder names for development phase')


    args = parser.parse_args()

    ###############################################################################""
    user_dictionnary, utterrance_dev, labels_dev, utterrance_test, labels_test = ExtractDataset(args.dev_folder_path, args.test_folder_path)


    labels_dev = labels_dev.reshape(labels_dev.shape[0],)
    labels_test = labels_test.reshape(labels_test.shape[0],)

    print("labels_dev {0} utterrance_dev {1}".format(labels_dev.shape,utterrance_dev.shape))
    utterrance_dev_copy = utterrance_dev.copy()
    labels_dev_copy = labels_dev.copy()
    idx = np.random.randint(labels_dev.shape[0], size=labels_dev.shape[0])
    print("idx {0}".format(idx))
    for num, index in enumerate(idx):
        print("num {0} index {1}".format(num,index))
        utterrance_dev_copy[num, :, :, :] = utterrance_dev[index, :, :,:]
        labels_dev[num] = labels_dev_copy[index]


    print("labels_test {0} utterrance_test {1}".format(labels_test.shape,utterrance_test.shape))
    utterrance_test_copy = utterrance_test.copy()
    labels_test_copy = labels_test.copy()
    idx = np.random.randint(labels_test.shape[0], size=labels_test.shape[0])
    print("idx {0}".format(idx))
    for num, index in enumerate(idx):
        print("num {0} index {1}".format(num,index))
        utterrance_test_copy[num, :, :, :] = utterrance_test[index, :, :,:]
        labels_test[num] = labels_test_copy[index]


    print('\n \n \n ---- END process')
    print(labels_test)

    print(labels_dev.shape)
    print(labels_test.shape)    
    print(utterrance_dev.shape)
    print(utterrance_test.shape)




    with h5py.File("development.hdf5", "w") as f:
        dset = f.create_dataset("label_test", data=labels_test)
        dset = f.create_dataset("label_train", data=labels_dev)
        dset = f.create_dataset("utterance_test", data=utterrance_test)
        dset = f.create_dataset("utterance_train", data=utterrance_dev)