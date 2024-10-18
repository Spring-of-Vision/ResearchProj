import os
import numpy as np

base_directory = '/content/drive/MyDrive/Colab Notebooks/first chunk'

total_files_count = 0
i = 0

for folder_name in ['GoogleDoc', 'GoogleDrive', 'Youtube']:
    folder_path = os.path.join(base_directory, folder_name)

    if os.path.isdir(folder_path):
        print(f"pass on : {folder_name}")
        files_count = len([f for f in os.listdir(folder_path) if f.endswith('.npy')])
        total_files_count += files_count

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(folder_path, file_name)
                file_class = file_name.split('-')[0]

                array = np.load(file_path)
                #print(array.dtype, array.shape)

                if(array[31][4] <= 0.009):
                    if(array[30][0] <= 0.055):
                        if(array[22][0]<=0.234):
                            if(array[5][0]<=0.265): classes = "Youtube"
                            else: classes = "GoogleDoc"
                        else: classes = "GoogleDrive"
                    else:
                        if(array[29][4]<=0.016): classes = "GoogleDrive"
                        else: classes = "GoogleDoc"
                else: classes = "GoogleDoc"

                print(file_class, classes)
                if (file_class==classes):
                  i+=1


print(f"{i}/{total_files_count} is {i/total_files_count} accuracy")