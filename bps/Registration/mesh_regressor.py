# path to the data and the mesh regressor model!
FAUST_DATA_PATH = '/home/khanm/workfolder/bps/Registration/data/MPI-FAUST/'
CKPT_DATA_PATH = '/home/khanm/workfolder/bps/Registration/data/mesh_regressor.h5'

# all the results will be saved here
LOG_DIR = '/home/khanm/workfolder/bps/Registration/logs'

import faust
from bps import bps

from mesh_regressor_model import MeshRegressorMLP
import torch

import os
import numpy as np
from chamfer_distance import chamfer_distance

from tqdm import tqdm

import PIL


BPS_RADIUS = 1.7
N_BPS_POINTS = 1024
MESH_SCALER = 1000

train_scans, train_meshes = faust.get_faust_train(FAUST_DATA_PATH)
test_scans = faust.get_faust_test(FAUST_DATA_PATH)

xtr, xtr_mean, xtr_max = bps.normalize(train_scans, max_rescale=False, return_scalers=True)
ytr = bps.normalize(train_meshes, x_mean=xtr_mean, x_max=xtr_max, known_scalers=True, max_rescale=False)
xtr_bps = bps.encode(xtr, radius=BPS_RADIUS, n_bps_points=N_BPS_POINTS, bps_cell_type='dists')

xte, xte_mean, xte_max = bps.normalize(test_scans, max_rescale=False, return_scalers=True)
xte_bps = bps.encode(xte, radius=BPS_RADIUS, n_bps_points=N_BPS_POINTS, bps_cell_type='dists')

model = MeshRegressorMLP(n_features=N_BPS_POINTS)
model.load_state_dict(torch.load(CKPT_DATA_PATH, map_location='cpu'))
model.eval()

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

yte_preds = model(torch.Tensor(xte_bps)).detach().numpy() 
yte_preds /= MESH_SCALER

scan2mesh_losses = []
for fid in range(0, len(yte_preds)):
    yte_preds[fid] += xte_mean[fid] #bring back center of the mass after prediction
    scan2mesh_losses.append(MESH_SCALER*chamfer_distance(yte_preds[fid], xte[fid]+xte_mean[fid], direction='y_to_x'))

alignments_path = os.path.join(LOG_DIR, 'alignments.npy')
np.save(alignments_path, yte_preds)

print("FAUST test set scan-to-mesh distance (avg): %4.2f mms" % np.mean(scan2mesh_losses))

N_TEST_SCANS = 200

smpl_faces = np.loadtxt('../bps_demos/smpl_mesh_faces.txt')

for sid in tqdm(range(0, N_TEST_SCANS)):
    
    mesh_verts = yte_preds[sid]
    scan2mesh_loss = scan2mesh_losses[sid]
    mesh_scan = faust.get_faust_scan_by_id(FAUST_DATA_PATH, sid, 'test')
    
    faust.visualise_predictions(scan=mesh_scan, align_verts=mesh_verts, align_faces=smpl_faces, 
                                scan_id=sid, scan2mesh_loss=scan2mesh_loss, save_dir=LOG_DIR)
                                

images_dir = os.path.join(LOG_DIR, 'faust_test_images')
aligns_dir =  os.path.join(LOG_DIR, 'faust_test_alignments')
print("Predicted alignments meshes saved in: %s" % images_dir)
print("Predicted alignments images saved in: %s" % aligns_dir)

fid = 0
merged_img_path =  os.path.join(images_dir, '%03d_scan_align_pair.png'%fid)
PIL.Image.open(merged_img_path)