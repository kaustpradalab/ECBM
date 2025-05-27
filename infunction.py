from CUB.dataset import load_data

from os import environ
import pytorch_influence_functions as ptif
import torch
import torchvision.transforms as tvtf
import json



def get_dataset_sample_ids_wrap(num_samples, test_loader, num_classes=None, start_index=0):
    assert start_index == 0
    assert num_classes is not None

    sample_dict = {str(i): list() for i in range(num_classes)}
    idx = -1
    for (_, label_batch) in test_loader:
        for label in label_batch:
            idx += 1

            label = str(int(label))
            if len(sample_dict[label]) < num_samples:
                sample_dict[label].append(idx)

    sample_list = [sample_dict[str(i)] for i in range(num_classes)]
    return sample_dict, sample_list

ptif.get_dataset_sample_ids = get_dataset_sample_ids_wrap


def main():
    ptif.init_logging()
    config = ptif.get_default_config()
    device = torch.zeros(1).device

    config['dataset'] = 'CUB'
    config['num_classes'] = 200
    config['outdir'] = 'outputs/infunction'

    train_data_path = environ['INFUNCTION_TRAIN_DATA_PATH'] #'./CUB_processed/class_attr_data_10/train.pkl'
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    model_path = environ['INFUNCTION_MODEL_PATH'] #'./outputs/42/Independent_CtoY/best_model.pth'
    model = torch.load(model_path).eval().to(device)
    loader_train = load_data([train_data_path], use_attr=True, no_img=True, batch_size=64)
    loader_val = load_data([val_data_path], use_attr=True, no_img=True, batch_size=64)

    _fw = model.forward
    model.forward = lambda *arg: _fw(arg[0].float())

    influences, harmful, helpful = ptif.calc_img_wise(config, model, loader_train, loader_val)

    with open('./influences.json', 'w') as file:
        json.dump(influences, file)
    with open('./harmful.json', 'w') as file:
        json.dump(harmful, file)
    with open('./helpful.json', 'w') as file:
        json.dump(helpful, file)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_device('cuda')
    main()
