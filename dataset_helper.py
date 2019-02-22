import numpy as np
import os
from shutil import copyfile

import urllib2
import zipfile

cwd = os.getcwd()

def download_post_processing(dataset_name):
    if dataset_name == "convex":
        zip_ref = zipfile.ZipFile(cwd + "/" + dataset_name + ".zip", 'r')
        zip_ref.extractall(cwd)
        zip_ref.close()
        print(dataset_name + " " + "extracted successfuly")
        copyfile(cwd + "/50k/convex_test.amat", cwd + "/convex_test.amat")
        print(dataset_name + " " + "post processed successfuly")
    elif dataset_name == "mnist-background":
        zip_ref = zipfile.ZipFile(cwd + "/" + dataset_name + ".zip", 'r')
        zip_ref.extractall(cwd)
        zip_ref.close()
        print(dataset_name + " " + "extracted successfuly")
    elif dataset_name == "mnist-random-background":
        zip_ref = zipfile.ZipFile(cwd + "/" + dataset_name + ".zip", 'r')
        zip_ref.extractall(cwd)
        zip_ref.close()
        print(dataset_name + " " + "extracted successfuly")
    elif dataset_name == "mnist-rotation":
        zip_ref = zipfile.ZipFile(cwd + "/" + dataset_name + ".zip", 'r')
        zip_ref.extractall(cwd)
        zip_ref.close()
        print(dataset_name + " " + "extracted successfuly")
    elif dataset_name == "rectangles":
        zip_ref = zipfile.ZipFile(cwd + "/" + dataset_name + ".zip", 'r')
        zip_ref.extractall(cwd)
        zip_ref.close()
        print(dataset_name + " " + "extracted successfuly")
    elif dataset_name == "mnist":
        zip_ref = zipfile.ZipFile(cwd + "/" + dataset_name + ".zip", 'r')
        zip_ref.extractall(cwd)
        zip_ref.close()
        print(dataset_name + " " + "extracted successfuly")


def download_and_unzip_file(dataset_name):
    if dataset_name == "convex":
        filedata = urllib2.urlopen('http://www.iro.umontreal.ca/~lisa/icml2007data/convex.zip')
        print(dataset_name + " " + "downloaded successfuly")
        datatowrite = filedata.read()
        with open(cwd + "/" + dataset_name + ".zip", 'wb') as f:
            f.write(datatowrite)
        download_post_processing(dataset_name)
    elif dataset_name == "mnist-background":
        filedata = urllib2.urlopen('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_images.zip')
        print(dataset_name + " " + "downloaded successfuly")
        datatowrite = filedata.read()
        with open(cwd + "/" + dataset_name + ".zip", 'wb') as f:
            f.write(datatowrite)
        download_post_processing(dataset_name)
    elif dataset_name == "mnist-random-background":
        filedata = urllib2.urlopen('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_background_random.zip')
        print(dataset_name + " " + "downloaded successfuly")
        datatowrite = filedata.read()
        with open(cwd + "/" + dataset_name + ".zip", 'wb') as f:
            f.write(datatowrite)
        download_post_processing(dataset_name)
    elif dataset_name == "mnist-rotation":
        filedata = urllib2.urlopen('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip')
        print(dataset_name + " " + "downloaded successfuly")
        datatowrite = filedata.read()
        with open(cwd + "/" + dataset_name + ".zip", 'wb') as f:
            f.write(datatowrite)
        download_post_processing(dataset_name)
    elif dataset_name == "rectangles":
        filedata = urllib2.urlopen('http://www.iro.umontreal.ca/~lisa/icml2007data/rectangles.zip')
        print(dataset_name + " " + "downloaded successfuly")
        datatowrite = filedata.read()
        with open(cwd + "/" + dataset_name + ".zip", 'wb') as f:
            f.write(datatowrite)
        download_post_processing(dataset_name)
    elif dataset_name == "mnist":
        filedata = urllib2.urlopen('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip')
        print(dataset_name + " " + "downloaded successfuly")
        datatowrite = filedata.read()
        with open(cwd + "/" + dataset_name + ".zip", 'wb') as f:
            f.write(datatowrite)
        download_post_processing(dataset_name)

def vectorize_labels(labels):
    vecorized_labels = []
    for i in labels:
        zero_array = np.zeros(10)
        zero_array[int(i[0])] = 1
        vecorized_labels.append(zero_array)
    return np.asarray(vecorized_labels)

def load_mnist(name):
    file_path = cwd + "/mnist_train.amat"
    if not os.path.isfile(file_path):
        download_and_unzip_file(name)
    traing_data = np.loadtxt(cwd + "/" + "mnist_train.amat", dtype='float32')
    train_x = traing_data[:, :-1]  # / 1.0
    train_y =  vectorize_labels(traing_data[:, -1:])
    test_data = np.loadtxt(cwd + "/" + "mnist_test.amat", dtype='float32')
    test_x = test_data[:, :-1]  # / 1.0
    test_y =  vectorize_labels(test_data[:, -1:])
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}

def load_convex(name):
    file_path = cwd + "/convex_train.amat"
    if not os.path.isfile(file_path):
        download_and_unzip_file(name)
    traing_data = np.loadtxt(cwd + "/" + "convex_train.amat", dtype='float32')
    train_x = traing_data[:, :-1]  # / 1.0
    train_y = traing_data[:, -1:]
    test_data = np.loadtxt(cwd + "/" + "convex_test.amat", dtype='float32')
    test_x = test_data[:, :-1]  # / 1.0
    test_y = test_data[:, -1:]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}


def load_mnist_background(name):
    file_path = cwd + "/mnist_background_images_train.amat"
    if not os.path.isfile(file_path):
        download_and_unzip_file(name)
    traing_data = np.loadtxt('mnist_background_images_train.amat', dtype='float32')
    train_x = traing_data[:, :-1]  # / 1.0
    train_y = vectorize_labels(traing_data[:, -1:])
    test_data = np.loadtxt('mnist_background_images_test.amat', dtype='float32')
    test_x = test_data[:, :-1]  # / 1.0
    test_y = vectorize_labels(test_data[:, -1:])
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}

def load_mnist_rotation(name):
    file_path = cwd + "/mnist_all_rotation_normalized_float_train_valid.amat"
    if not os.path.isfile(file_path):
        download_and_unzip_file(name)
    traing_data = np.loadtxt('mnist_all_rotation_normalized_float_train_valid.amat', dtype='float32')
    train_x = traing_data[:, :-1]  # / 1.0
    train_y = vectorize_labels(traing_data[:, -1:])
    test_data = np.loadtxt('mnist_all_rotation_normalized_float_test.amat', dtype='float32')
    test_x = test_data[:, :-1]  # / 1.0
    test_y = vectorize_labels(test_data[:, -1:])
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}

def load_mnist_random_background(name):
    file_path = cwd + "/mnist_background_random_train.amat"
    if not os.path.isfile(file_path):
        download_and_unzip_file(name)
    traing_data = np.loadtxt(cwd + "/" + "mnist_background_random_train.amat", dtype='float32')
    train_x = traing_data[:, :-1]  # / 1.0
    train_y = vectorize_labels(traing_data[:, -1:])
    test_data = np.loadtxt(cwd + "/" + "mnist_background_random_test.amat", dtype='float32')
    test_x = test_data[:, :-1]  # / 1.0
    test_y = vectorize_labels(test_data[:, -1:])
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}

def load_rectangles(name):
    file_path = cwd + "/rectangles_train.amat"
    if not os.path.isfile(file_path):
        download_and_unzip_file(name)
    traing_data = np.loadtxt(cwd + "/" + "rectangles_train.amat", dtype='float32')
    train_x = traing_data[:, :-1] #/ 1.0
    train_y = traing_data[:, -1:]
    test_data = np.loadtxt(cwd + "/" + "rectangles_test.amat", dtype='float32')
    test_x = test_data[:, :-1] #/ 1.0
    test_y = test_data[:, -1:]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}

def load_nus_wide(name):
    traing_data = np.loadtxt(cwd + "/" + "Train_EDH.txt", dtype='float32')
    train_x = traing_data[:, :-1] #/ 1.0
    train_y = traing_data[:, -1:]
    test_data = np.loadtxt(cwd + "/" + "Test_WT.txt", dtype='float32')
    test_x = test_data[:, :-1] #/ 1.0
    test_y = test_data[:, -1:]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}




def load_dataset_by_name(name):
    if name == "mnist":
        return load_mnist(name)
    elif name == "NUS-WIDE":
        return load_nus_wide(name)
    elif name == "rectangles":
        return load_rectangles(name)
    elif name == "mnist-background":
        return load_mnist_background(name)
    elif name == "convex":
        return load_convex(name)
    elif name == "mnist-rotation":
        return load_mnist_rotation(name)
    elif name == "mnist-random-background":
        return load_mnist_random_background(name)
    else:
        raise ValueError("No Dataset with this name")
