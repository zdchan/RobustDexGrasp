from shutil import copyfile
import datetime
import os
import ntpath
import torch


class ConfigurationSaver:
    def __init__(self, log_dir, save_items, test_dir=False, task_name=''):
        if test_dir:
            self._data_dir = log_dir + '/' + "eval/" + task_name + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            self._data_dir = log_dir + '/' + task_name + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


        if not os.path.isdir(self._data_dir):
            os.makedirs(self._data_dir)

        if save_items is not None:
            for save_item in save_items:
                base_file_name = ntpath.basename(save_item)
                copyfile(save_item, self._data_dir + '/' + base_file_name)

    @property
    def data_dir(self):
        return self._data_dir
        

def tensorboard_launcher(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)


def load_param(weight_path, env, actor, critic, optimizer, data_dir, cfg):
    if weight_path == "":
        raise Exception("\nCan't find the pre-trained weight, please provide a pre-trained weight with --weight switch\n")
    print("\nloading from the checkpoint:", weight_path+"\n")

    iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0].split('_')[0]
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'

    # mean_csv_path = weight_dir + 'mean_r' + iteration_number + '.csv'
    # var_csv_path = weight_dir + 'var_r' + iteration_number + '.csv'
    #items_to_save = [weight_path, mean_csv_path, var_csv_path, weight_dir + cfg, weight_dir + "Environment.hpp"]
    items_to_save = []

    if items_to_save is not None:
        pretrained_data_dir = data_dir + '/pretrained_' + weight_path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        if not os.path.isdir(pretrained_data_dir):
            os.makedirs(pretrained_data_dir)
        for item_to_save in items_to_save:
            # print(pretrained_data_dir)
            # print(item_to_save.rsplit('/', 1)[1])
            # print(item_to_save)
            copyfile(item_to_save, pretrained_data_dir+'/'+item_to_save.rsplit('/', 1)[1])

    # load observation scaling from files of pre-trained model
    env.load_scaling(weight_dir, iteration_number)

    # load actor and critic parameters from full checkpoint
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu') )
    actor.architecture.load_state_dict(checkpoint['actor_architecture_state_dict'])
    actor.distribution.load_state_dict(checkpoint['actor_distribution_state_dict'])
    critic.architecture.load_state_dict(checkpoint['critic_architecture_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
