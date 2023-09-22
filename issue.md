## Issues of generating ground truth poses in former repos

In `preprocess/generate_kitti_pairs.py`, we follow [FCGF](https://github.com/chrischoy/FCGF), [PREDATOR](https://github.com/prs-eth/OverlapPredator), [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences), [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), and [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) to refine the ground truth relative poses using ICP. However, we find a same issue occurred in these repos, which may originates from  [FCGF](https://github.com/chrischoy/FCGF). 

Take [FCGF](https://github.com/chrischoy/FCGF) as example. When generating ground truth pairs, it follows (line 456-467 in `lib/data_loaders.py`):

```python
M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
             @ np.linalg.inv(self.velo2cam)).T
xyz0_t = self.apply_transform(xyz0[sel0], M)
pcd0 = make_open3d_point_cloud(xyz0_t)
pcd1 = make_open3d_point_cloud(xyz1[sel1])
reg = o3d.pipelines.registration.registration_icp(
pcd0, pcd1, 0.2, np.eye(4),
o3d.pipelines.registration.TransformationEstimationPointToPoint(),
o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
pcd0.transform(reg.transformation)
# pcd0.transform(M2) or self.apply_transform(xyz0, M2)
M2 = M @ reg.transformation
```

Above code first calculates the relative pose `M` from scan `xyz1` to scan `xyz0` using ground truth poses. Due to the issues in ground truth pose of KITTI, there can be a significant drift between `xyz1` and transformed `xyz0_t`, where `xyz0_t` is transformed `xyz0` using `M`. We denote `M` as T_0t_to_0 referring to the transformation from scan `xyz0_t` to scan `xyz0_t`. It then calculates the pose drift `reg.transformation` from scan `xyz0_t` to scan `xyz1` using ICP, denoted as T_1_to_0t. The refined pose `M2` should be T_1_to_0t * T_0t_to_0, or ` reg.transformation @ M`.

