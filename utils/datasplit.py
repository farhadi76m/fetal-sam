from sklearn.model_selection import train_test_split
import json
import os

def split_data(data_path, data_ratio = [0.70, 0.15, 0.15]):
    data = os.listdir(data_path)

    train_data, temp_data = train_test_split(data,
                                            test_size = (data_ratio[1] + data_ratio[2]),
                                            random_state=42
                                            )
    val_data, test_data = train_test_split(temp_data,
                                        test_size=data_ratio[2],
                                        random_state=42
                                        )

    for data in [train_data, val_data, test_data]:
        for i, v in enumerate(data):
            data[i]= {'image': '/images/'+ v,
                    'label': {'color': '/labels/' + os.path.splitext(v)[0] + '_gtFine_color.png',
                                'instanceIds': '/labels/' + os.path.splitext(v)[0] + '_gtFine_instanceIds.png',
                                'labelIds': '/labels/' + os.path.splitext(v)[0] + '_gtFine_labelIds.png'}
                    }
    dataset = {'train': train_data,
            'validation': val_data,
            'test': test_data}

    with open('dataset.json', 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

if __name__ == "__main__":

    data_path = "<your data path>"
    split_data(data_path)