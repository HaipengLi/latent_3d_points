import os
import os.path as osp
import sys
import numpy as np
import scipy.io as sio

sys.path.append('../..')
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# Choose the GPU
os.environ["CUDA_VISIBLE_DEVICES"]='1'

top_out_dir = '../output/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../../latent_3d_points_data/' # Top-dir of where point-clouds are stored.

class_name_A = 'chair'
class_name_B = 'table'

experiment_name = 'single_class_ae_{}_{}'.format(class_name_A, class_name_B)
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'

datafolder = top_in_dir + class_name_A + '-' + class_name_B + '/'
train_dir_A = datafolder + class_name_A + '_train'
train_dir_B = datafolder + class_name_B + '_train'
test_dir_A =  datafolder + class_name_A + '_test'
test_dir_B =  datafolder+ class_name_B + '_test'

# all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
training_pc_data_A = load_all_point_clouds_under_folder(train_dir_A, n_threads=8, file_ending='.ply', verbose=True)
training_pc_data_B = load_all_point_clouds_under_folder(train_dir_B, n_threads=8, file_ending='.ply', verbose=True)

testing_pc_data_A = load_all_point_clouds_under_folder(test_dir_A, n_threads=8, file_ending='.ply', verbose=True)
testing_pc_data_B = load_all_point_clouds_under_folder(test_dir_B, n_threads=8, file_ending='.ply', verbose=True)

all_pc_data = training_pc_data_A
all_pc_data.merge(training_pc_data_B)
all_pc_data.shuffle_data()

all_pc_data_test = testing_pc_data_A
all_pc_data_test.merge(testing_pc_data_B)

train_params = default_train_params()

encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on 
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))

load_pre_trained_ae = True
restore_epoch = 500
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)
else:
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
# fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
# train_stats = ae.train(all_pc_data, conf, log_file=fout)
# fout.close()

NUM_TEST = 10
feed_pc, feed_model_names, _ = all_pc_data_test.next_batch(NUM_TEST)
reconstructions = ae.reconstruct(feed_pc)[0]
latent_codes = ae.transform(feed_pc)

# for i in range(NUM_TEST):
#     plot_3d_point_cloud(reconstructions[i][:, 0], 
#                         reconstructions[i][:, 1], 
#                         reconstructions[i][:, 2], in_u_sphere=True)

# Save the latent code
training_pc_data_A = load_all_point_clouds_under_folder(train_dir_A, n_threads=8, file_ending='.ply', verbose=True)
training_pc_data_B = load_all_point_clouds_under_folder(train_dir_B, n_threads=8, file_ending='.ply', verbose=True)

def pc_to_latent(ae, pc_data):
    batch_size = 32
    batches = int(all_pc_data_test.num_examples / batch_size)
    latent_code_list = None
    for _ in range(batches):
        feed_pc, _, _ = pc_data.next_batch(batch_size)
        latent_codes = ae.transform(feed_pc)
        latent_code_list = (np.vstack([latent_code_list, latent_codes])
                            if latent_code_list is not None else latent_codes)

    return latent_code_list

latent_A = pc_to_latent(ae, training_pc_data_A)
latent_B = pc_to_latent(ae, training_pc_data_B)

latent_out_file = os.path.join(train_dir, 'latent-{}-{}.mat'.format(class_name_A, class_name_B))

sio.savemat(latent_out_file, {'latentcodes_A': latent_A ,  'latentcodes_B': latent_B })