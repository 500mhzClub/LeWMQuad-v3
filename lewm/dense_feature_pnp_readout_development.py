"""Fixed geometric motion readout from current depth and dense feature matches."""
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, INTRINSICS

SETTINGS = dict(feature_grid_hw=[24,32],native_patch_width_px=20,
    matcher='mutual nearest cosine neighbours',minimum_matches=12,
    ransac_reprojection_error_px=12.,ransac_iterations=300,ransac_confidence=.999,
    ransac_seed=2026091804,depth_central_patch_width_px=4,
    maximum_local_depth_spread_absolute_m=.02,maximum_local_depth_spread_fraction=.05)


@torch.inference_mode()
def decode(current, future, depth):
    if current.shape!=(768,1024) or future.shape!=current.shape:
        raise ValueError('two dense 24x32 feature grids required')
    if not torch.isfinite(current).all() or not torch.isfinite(future).all():
        raise ValueError('finite features required')
    values = np.asarray(depth['depth_m']); valid = np.asarray(depth['valid'])
    if values.shape!=(480,640) or valid.shape!=values.shape:
        raise ValueError('current native aligned optical depth required')
    similarity = F.normalize(current.float(),dim=-1)@F.normalize(future.float(),dim=-1).T
    target = similarity.argmax(1)
    reverse = similarity.argmax(0)
    index = torch.arange(768,device=current.device)
    mutual = (reverse[target]==index).cpu().numpy()
    target = target.cpu().numpy()
    xy = np.stack(np.meshgrid(np.arange(32)*20+10,np.arange(24)*20+10),axis=-1).reshape(-1,2).astype(np.float64)
    source_ids = []; optical = []; pixels = []
    K = np.asarray(INTRINSICS,dtype=np.float64)
    for i in np.flatnonzero(mutual):
        u,v = xy[i].astype(int)
        patch = values[v-2:v+2,u-2:u+2]; mask = valid[v-2:v+2,u-2:u+2]
        z = patch[mask]
        if len(z)<12:
            continue
        middle = float(np.median(z))
        if float(z.max()-z.min())>max(.02,.05*middle):
            continue
        source_ids.append(int(i))
        optical.append([(u-K[0,2])*middle/K[0,0],(v-K[1,2])*middle/K[1,1],middle])
        pixels.append(xy[target[i]])
    record = dict(mutual_matches=int(mutual.sum()),depth_supported_matches=len(optical),
        valid=False,motion_xy_yaw=None,current_depth_only=True,
        learned_readout=False,future_depth_used=False)
    if len(optical)<12:
        return record|dict(reason='INSUFFICIENT_MATCHES')
    points = np.asarray(optical,dtype=np.float64)
    pixels = np.asarray(pixels,dtype=np.float64)
    cv2.setRNGSeed(SETTINGS['ransac_seed'])
    success,rvec,tvec,inliers = cv2.solvePnPRansac(points,pixels,K,None,
        iterationsCount=300,reprojectionError=12.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    count = 0 if inliers is None else len(inliers)
    record.update(inliers=count)
    if not success or count<12:
        return record|dict(reason='INSUFFICIENT_PNP_INLIERS')
    chosen = inliers[:,0]
    rvec,tvec = cv2.solvePnPRefineLM(points[chosen],pixels[chosen],K,None,rvec,tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    projected = cv2.projectPoints(points[chosen],rvec,tvec,K,None)[0].reshape(-1,2)
    errors = np.linalg.norm(projected-pixels[chosen],axis=1)
    transformed = points[chosen]@rotation.T+tvec.reshape(3)
    if not np.isfinite(transformed).all() or np.any(transformed[:,2]<=0):
        return record|dict(reason='INVALID_CAMERA_POSE')
    future_from_current = np.eye(4)
    future_from_current[:3,:3] = rotation; future_from_current[:3,3] = tvec.reshape(3)
    mount = np.asarray(BODY_FROM_OPTICAL,dtype=np.float64)
    body = mount@np.linalg.inv(future_from_current)@np.linalg.inv(mount)
    yaw = float(np.arctan2(body[1,0],body[0,0]))
    return record|dict(valid=True,reason=None,motion_xy_yaw=[float(body[0,3]),float(body[1,3]),yaw],
        rotation_initial_from_future=body[:3,:3].tolist(),translation_initial_body_m=body[:3,3].tolist(),
        median_inlier_reprojection_px=float(np.median(errors)),max_inlier_reprojection_px=float(errors.max()))
