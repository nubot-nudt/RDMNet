import numpy as np
import open3d as o3d
import torch
import pykitti

# import teaserpp_python
def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val**2 * n
        self.var = self.sq_sum / self.count - self.avg**2

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)

# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print('\t'.join(entries))

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = '{:' + str(num_digits) + 'd}'
#         return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def evaluate_registration(mkpts0, mkpts1, pred, b, T_gt, method='svd'):
    '''calculate pose'''
    if method == 'ransac':
        kpts0, kpts1 = pred['keypoints0'][b].cpu().numpy()[:,:3], pred['keypoints1'][b].cpu().numpy()[:,:3]
        feat0, feat1 = pred['descriptors0'][b].cpu().numpy(), pred['descriptors1'][b].cpu().numpy()
        feat0, feat1 = pred['mdesc0'][b].cpu().numpy(), pred['mdesc1'][b].cpu().numpy()
        src_features = o3d.pipelines.registration.Feature()
        src_features.data = feat1.T
        src_keypts = o3d.geometry.PointCloud()
        src_keypts.points = o3d.utility.Vector3dVector(kpts1)

        tgt_features = o3d.pipelines.registration.Feature()
        tgt_features.data = feat0.T
        tgt_keypts = o3d.geometry.PointCloud()
        tgt_keypts.points = o3d.utility.Vector3dVector(kpts0)

        result_ransac  = execute_global_registration(src_keypts, tgt_keypts, src_features, tgt_features ,  0.5)
        T = torch.tensor(result_ransac.transformation, dtype=torch.double)
    elif method == 'svd':
        T = solve_icp(mkpts0, mkpts1)
        # T = solve_teaser(mkpts1, mkpts0)
        T = torch.tensor(T, dtype=torch.double)
        
    '''calculate inlier'''
    if len(mkpts1)>0:
        kp0_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in mkpts0])
        kp1_np = np.array([(kp[0], kp[1], kp[2], 1) for kp in mkpts1])
        mkpts0 = torch.tensor(kp0_np, dtype=torch.double)
        mkpts1 = torch.tensor(kp1_np, dtype=torch.double)
        mkpts0w = torch.einsum('ki,ij->jk', T_gt, mkpts0.T)
        inlier = torch.norm(mkpts0w[:,:3] - mkpts1[:,:3], dim=1) < 0.5
        # mkpts1w = torch.einsum('ki,ij->jk', T_gt, mkpts1.T)
        # inlier = torch.norm(mkpts1w[:,:3] - mkpts0[:,:3], dim=1) < 0.5
        inlier = inlier.sum()
        inlier_ratio = inlier.item()/len(mkpts0)
    else:
        inlier = 0
        inlier_ratio = 0
    # print(inlier,' ',len(mkpts0))

    '''calculate rre rte'''
    T_error = torch.einsum('ab,bc->ac', torch.inverse(T), T_gt).numpy()
    rte = np.linalg.norm(T_error[:3, 3])
    f_theta = (T_error[0, 0] + T_error[1, 1] + T_error[2, 2] -1) /2
    f_theta = max(min(f_theta, 1), -1)  
    rre = np.arccos(f_theta)
    rre = 180 * rre / np.pi
    return  T, rte, rre, inlier, inlier_ratio

def solve_icp(P, Q):
    """
    Solve ICP

    Parameters
    ----------
    P: numpy.ndarray
        source point cloud as N-by-3 numpy.ndarray
    Q: numpy.ndarray
        target point cloud as N-by-3 numpy.ndarray

    Returns
    ----------
    T: transform matrix as 4-by-4 numpy.ndarray
        transformation matrix from one-step ICP

    """
    # compute centers:
    up = P.mean(axis = 0)
    uq = Q.mean(axis = 0)

    # move to center:
    P_centered = P - up
    Q_centered = Q - uq
    # P_centered = P 
    # Q_centered = Q 

    U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = uq - np.dot(R, up)

    # format as transform:
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0

    return T

def solve_teaser(P, Q):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.01
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    # print("Parameters are:", solver_params)
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(P.T, Q.T)

    solution = solver.getSolution()


    T = np.zeros((4, 4))
    T[0:3, 0:3] = solution.rotation
    T[0:3, 3] = solution.translation
    T[3, 3] = 1.0
    return T

def point2inch(x):
    return x / 72.

def plot_registration(pc0, pc1, T, T_gt=None, T_gt_gps=None):
    pc0 = pc0[pc0[:,2]>-5]
    pc1 = pc1[pc1[:,2]>-5]

    if np.isnan(T.cpu().numpy()).sum()>0:
        pcd_source_keypoints, pcd_target_keypoints = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        # pcd_source_keypoints=[]; pcd_target_keypoints=[]
    else:
        pcd_source_keypoints, pcd_target_keypoints = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        pcd_source_keypoints.points, pcd_target_keypoints.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1[:,:3])
        pcd_source_keypoints.paint_uniform_color([1.0, 0.0, 0.0]), pcd_target_keypoints.paint_uniform_color([0.0, 1.0, 0.0])
        pcd_source_keypoints.transform(T)


    if T_gt is not None:
        pcd_source_keypoints2, pcd_target_keypoints2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        pcd_source_keypoints2.points, pcd_target_keypoints2.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1[:,:3])
        pcd_source_keypoints2.paint_uniform_color([1.0, 0.0, 0.0]), pcd_target_keypoints2.paint_uniform_color([0.0, 1.0, 0.0])
        pcd_source_keypoints2.transform(T_gt)
        pcd_source_keypoints2.translate([0,100,0]), pcd_target_keypoints2.translate([0,100,0])

        pcd_source_keypoints3, pcd_target_keypoints3 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        pcd_source_keypoints3.points, pcd_target_keypoints3.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1[:,:3])
        pcd_source_keypoints3.paint_uniform_color([1.0, 0.0, 0.0]), pcd_target_keypoints3.paint_uniform_color([0.0, 1.0, 0.0])
        pcd_source_keypoints3.transform(T_gt_gps)
        pcd_source_keypoints3.translate([0,200,0]), pcd_target_keypoints3.translate([0,200,0])
        o3d.visualization.draw_geometries([pcd_source_keypoints+pcd_target_keypoints+pcd_source_keypoints2+pcd_target_keypoints2+pcd_target_keypoints3+pcd_source_keypoints3])
    else:
        o3d.visualization.draw_geometries([pcd_source_keypoints+pcd_target_keypoints])

def plot_match(pc0, pc1, kpts0, kpts1, mkpts0, mkpts1, mkpts0_gt, mkpts1_gt, matches, mconf, true_positive, false_positive, T, radius):
    """
        Visualize the matches
        from left to right: 
            all the matches; 
            all the matches visualized acoording to true matches and false matches;
            ground-truth matches
    """
    pc0 = pc0[pc0[:,2]>-5]
    pc1 = pc1[pc1[:,2]>-5]
    # pc0 = mkpts0[mkpts0[:,2]>-4]
    # pc1 = mkpts1[mkpts1[:,2]>-4]
    pcd_source_keypoints, pcd_target_keypoints = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    # pc1_np = np.array([(pc[0], pc[1], pc[2], 1) for pc in pc1])
    # pc1_torch = torch.tensor(pc1_np, dtype=torch.double)
    # pc1_registered = torch.einsum('ki,ij->jk', T, pc1_torch.T)
    # pc1_registered = pc1_registered[:, :3]
    # pcd_source_keypoints.points, pcd_target_keypoints.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1_registered.numpy())
    # pcd_source_keypoints.paint_uniform_color([0.15, 0.8, 0.15])
    # pcd_target_keypoints.paint_uniform_color([0.15, 0.15, 0.8])
    # o3d.visualization.draw_geometries([
    #                                     pcd_source_keypoints, pcd_target_keypoints])


    pcd_source_keypoints.points, pcd_target_keypoints.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1[:,:3])
    pcd_source_keypoints2, pcd_target_keypoints2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_source_keypoints2.points, pcd_target_keypoints2.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1[:,:3])
    pcd_source_keypoints3, pcd_target_keypoints3 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_source_keypoints3.points, pcd_target_keypoints3.points = o3d.utility.Vector3dVector(pc0[:,:3]), o3d.utility.Vector3dVector(pc1[:,:3])

    translation = np.asarray([0, 45, 0])
    pcd_source_keypoints_shifted = pcd_source_keypoints.translate(translation)
    pcd_target_keypoints_shifted = pcd_target_keypoints.translate(-translation)
    translation2 = np.asarray([300, 0, 0])
    pcd_source_keypoints_shifted2 = pcd_source_keypoints2.translate(translation + translation2)
    pcd_target_keypoints_shifted2 = pcd_target_keypoints2.translate(-translation + translation2)
    translation3 = np.asarray([150, 0, 0])
    pcd_source_keypoints_shifted3 = pcd_source_keypoints3.translate(translation + translation3)
    pcd_target_keypoints_shifted3 = pcd_target_keypoints3.translate(-translation + translation3)
    # mkpts0, mkpts1 = mkpts0[:,0,:], mkpts1[:,0,:]
    points = np.vstack((mkpts0 + translation, mkpts1 - translation))
    points2 = np.vstack((mkpts0_gt + translation +  translation2, mkpts1_gt - translation + translation2))
    t_kpts0, t_kpts1 = kpts0[true_positive], kpts1[matches[true_positive]]
    n_kpts0, n_kpts1 = kpts0[false_positive], kpts1[matches[false_positive]]
    
    t_points3 = np.vstack((t_kpts0 + translation +  translation3, t_kpts1 - translation + translation3))
    n_points3 = np.vstack((n_kpts0 + translation +  translation3, n_kpts1 - translation + translation3))


    pcd_keypoints, pcd_keypoints2, pcd_keypoints3, pcd_keypoints4 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pcd_keypoints.points, pcd_keypoints2.points = o3d.utility.Vector3dVector(points[:,:3]), o3d.utility.Vector3dVector(points2[:,:3])
    pcd_keypoints3.points, pcd_keypoints4.points = o3d.utility.Vector3dVector(t_points3[:,:3]), o3d.utility.Vector3dVector(n_points3[:,:3])

    N, _ = mkpts0.shape
    N2, _ = mkpts0_gt.shape
    N3 = np.sum(true_positive)
    N4 = np.sum(false_positive)
    correspondences, correspondences2 =([[i, i + N] for i in np.arange(N)]), ([[i, i + N2] for i in np.arange(N2)])
    correspondences3, correspondences4 =([[i, i + N3] for i in np.arange(N3)]), ([[i, i + N4] for i in np.arange(N4)])
    # colors = [[1-i, i, 0.2] for i in mconf]
    colors = [[1, 0, 0.2] for i in mconf]
    colors2 = [[0, 1, 0.2] for i in range(N2)]
    t_colors = [[0, 1, 0.2] for i in range(N3)]
    n_colors = [[1, 0, 0.2] for i in range(N4)]
    correspondence_set = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points),
        lines = o3d.utility.Vector2iVector(correspondences),
    )
    correspondence_set2 = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(points2),
        lines = o3d.utility.Vector2iVector(correspondences2),
    )
    correspondence_set3 = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(t_points3),
        lines = o3d.utility.Vector2iVector(correspondences3),
    )
    correspondence_set4 = o3d.geometry.LineSet(
        points = o3d.utility.Vector3dVector(n_points3),
        lines = o3d.utility.Vector2iVector(correspondences4),
    )

    # pcd_source_keypoints_shifted.paint_uniform_color([0.15, 0.15, 0.8]) 
    # pcd_source_keypoints_shifted2.paint_uniform_color([0.15, 0.8, 0.15]), pcd_source_keypoints_shifted3.paint_uniform_color([0.15, 0.15, 0.8])
    # pcd_target_keypoints_shifted.paint_uniform_color([0.15, 0.15, 0.8]) 
    # pcd_target_keypoints_shifted2.paint_uniform_color([0.15, 0.8, 0.15]), pcd_target_keypoints_shifted3.paint_uniform_color([0.15, 0.15, 0.8])
    pcd_keypoints.paint_uniform_color([1.0, 0.0, 0.0]), pcd_keypoints2.paint_uniform_color([1.0, 0.0, 0.0])
    pcd_keypoints3.paint_uniform_color([1.0, 0.0, 0.0]), pcd_keypoints4.paint_uniform_color([1.0, 0.0, 0.0])

    
    ## draw with line, cannot change line width
    # o3d.visualization.draw_geometries([
    #                                     pcd_source_keypoints_shifted, correspondence_set, pcd_keypoints, pcd_target_keypoints_shifted, 
    #                                     pcd_source_keypoints_shifted2, correspondence_set2, pcd_keypoints2, pcd_target_keypoints_shifted2, 
    #                                     pcd_source_keypoints_shifted3, correspondence_set3, correspondence_set4, pcd_keypoints3, pcd_keypoints4,  pcd_target_keypoints_shifted3, 
    #                                     ])

    correspondence_mesh = LineMesh(np.array(points), correspondences, colors, radius=radius)
    correspondence_mesh2 = LineMesh(np.array(points2), correspondences2, colors2, radius=radius)
    correspondence_mesh3 = LineMesh(np.array(t_points3), correspondences3, t_colors, radius=radius)
    correspondence_mesh4 = LineMesh(np.array(n_points3), correspondences4, n_colors, radius=radius)
    line_mesh_geoms = correspondence_mesh.cylinder_segments
    line_mesh2_geoms = correspondence_mesh2.cylinder_segments
    line_mesh3_geoms = correspondence_mesh3.cylinder_segments
    line_mesh4_geoms = correspondence_mesh4.cylinder_segments

    ## draw keypoints with sphere
    radius = 0.8
    box_list1 = []
    for i in range(kpts0.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(kpts0[i].reshape([3, 1]))
        mesh_box.translate(translation)
        mesh_box.paint_uniform_color([1, 0, 0])

        mesh_box2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box2.translate(kpts0[i].reshape([3, 1]))
        mesh_box2.translate(translation+translation2)
        mesh_box2.paint_uniform_color([1, 0, 0])

        mesh_box3 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box3.translate(kpts0[i].reshape([3, 1]))
        mesh_box3.translate(translation+translation3)
        mesh_box3.paint_uniform_color([1, 0, 0])
        box_list1.append(mesh_box)
        box_list1.append(mesh_box2)
        box_list1.append(mesh_box3)
    
    box_list2 = []
    for i in range(kpts1.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(kpts1[i].reshape([3, 1]))
        mesh_box.translate(-translation)
        mesh_box.paint_uniform_color([1, 0, 0])

        mesh_box2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box2.translate(kpts1[i].reshape([3, 1]))
        mesh_box2.translate(-translation+translation2)
        mesh_box2.paint_uniform_color([1, 0, 0])

        mesh_box3 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box3.translate(kpts1[i].reshape([3, 1]))
        mesh_box3.translate(-translation+translation3)
        mesh_box3.paint_uniform_color([1, 0, 0])
        box_list2.append(mesh_box)
        box_list2.append(mesh_box2)
        box_list2.append(mesh_box3)
    
    ## draw line with triangular mesh
    o3d.visualization.draw_geometries([
                                        pcd_source_keypoints_shifted, 
                                        # *line_mesh_geoms, 
                                        pcd_keypoints, pcd_target_keypoints_shifted, 
                                        pcd_source_keypoints_shifted2, *line_mesh2_geoms, pcd_keypoints2, pcd_target_keypoints_shifted2, 
                                        pcd_source_keypoints_shifted3, *line_mesh3_geoms, *line_mesh4_geoms, pcd_keypoints3, pcd_keypoints4,  pcd_target_keypoints_shifted3, 
                                        ]+box_list1+ box_list2)
    
    
    # o3d.visualization.draw_geometries([
    #                                     pcd_source_keypoints_shifted3, pcd_target_keypoints_shifted3
    #                                     ]+box_list1+ box_list2)



"""Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT

"""

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=np.array(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)
