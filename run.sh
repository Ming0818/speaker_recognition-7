
    development_dataset='/home/xavier/Desktop/developpement/Network/speaker_recognition/development.hdf5'


    python3.5 -u ./1-development/train_softmax.py --num_epochs=1 --batch_size=3 --development_dataset_path=$development_dataset --train_dir=results/TRAIN_CNN_3D/train_logs

