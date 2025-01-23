import os
import numpy as np
import pydicom
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# To resize images
def resize_image(image_path):
    dicom = pydicom.dcmread(image_path)

    # Convert the DCM image to a numpy array
    image_array = dicom.pixel_array

    # Resize the image using OpenCV
    resized_image = cv2.resize(image_array, (224, 224))

    return resized_image

# To stack on z-axis and pad tenors
def pad_to_3d(image_dict):

    grouped_arrays = {}
    for key, value in image_dict.items():
        if key[0] not in grouped_arrays:
            grouped_arrays[key[0]] = [value]
        else:
            grouped_arrays[key[0]].append(value)

    stacked_tensors = {}
    for key, values in grouped_arrays.items():
        stacked_tensors[key] = torch.from_numpy(np.stack(values, axis=0))

    for key, tensor in stacked_tensors.items():
        if 6-tensor.shape[0] > 0:
            pad_size = 6 - tensor.shape[0]
            stacked_tensors[key] = F.pad(tensor, (0, 0, 0, 0, pad_size, 0), 'constant', 0)

    return stacked_tensors

# To prepare for CNN
def prepare_for_cnn(root_directory, targets_df, study_id_chunk, exclude_array, batch_size=2):

    # Prepare features
    image_dict = {}
    max_shape = tuple([43008, 224])
    for study_id in study_id_chunk:
        if study_id not in exclude_array:
            series_id_list = os.listdir(f'{root_directory}/{study_id}')
            for series_id in series_id_list:
                images_list = os.listdir(f'{root_directory}/{study_id}/{series_id}')

                # Initialize list to store image arrays for the series
                image_arrays = []
                # Iterate over DICOM files in the series folder
                for idx in range(1, len(images_list)+1):
                    image_path = f'{root_directory}/{study_id}/{series_id}/{idx}.dcm'
                    resized_image = resize_image(image_path)
                    # Append resized_image array to the list
                    image_arrays.append(resized_image)

                # Vertically stack DCM images
                stacked_images = np.vstack(image_arrays)
                # Store stacked images as a NumPy array
                np_array = np.array(stacked_images)

                # Pad them
                padding = max_shape[0] - np_array.shape[0]
                if padding > 0:
                    padding_shape = ((0, padding), (0, 0))
                    padded_np_array = np.pad(np_array, padding_shape, mode='constant', constant_values=0) # 0 is black (If I'm not mistaken)
                    # Store image arrays in the dictionary with (study_id, series_id) tuple as key
                    image_dict[(study_id, series_id)] = padded_np_array
                else:
                    image_dict[(study_id, series_id)] = np_array
    stacked_tensors = pad_to_3d(image_dict)

    # Prepare targets
    targets_tensors = {}
    for key, _ in stacked_tensors.items():
        target = targets_df[targets_df['study_id'] == int(key)]
        transposed_df = target.iloc[:, 1:].T
        one_hot_array = []
        for _, row in transposed_df.iterrows():
            if row.values == 'Normal/Mild':
                one_hot_array.append([1, 0, 0])
            elif row.values == 'Moderate':
                one_hot_array.append([0, 1, 0])
            elif row.values == 'Severe':
                one_hot_array.append([0, 0, 1])
        targets_tensors[key] = torch.tensor(one_hot_array)

    # Convert all tensors to float32
    feature_tensors = [tensor.float() for tensor in stacked_tensors.values()]
    target_tensors = [tensor.float() for tensor in targets_tensors.values()]

    # Stack the tensors
    X_train = torch.stack(feature_tensors)
    scaled_tensor = X_train / X_train.max()
    y_train = torch.stack(target_tensors)   

    # Make torch DataLoader
    train_dataset = TensorDataset(scaled_tensor, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader