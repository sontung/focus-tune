#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.
import open3d as o3d
import faiss
import argparse
import logging
import math
import joblib
import time
from distutils.util import strtobool
from pathlib import Path
import poselib
import cv2
import numpy as np
import torch
from pykdtree.kdtree import KDTree
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ace_network import Regressor
from ace_util import get_pixel_grid, read_nvm_file
from dataset import CamLocDataset


_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def localize_pose_lib(pairs, f, c1, c2, max_error=16.0):
    """
    using pose lib to compute (usually best)
    """
    camera = {
        "model": "SIMPLE_PINHOLE",
        "height": int(c1 * 2),
        "width": int(c2 * 2),
        "params": [f, c1, c2],
    }
    object_points = []
    image_points = []
    for xy, xyz in pairs:
        xyz = np.array(xyz).reshape((3, 1))
        xy = np.array(xy)
        xy = xy.reshape((2, 1)).astype(np.float64)
        image_points.append(xy)
        object_points.append(xyz)
    pose, info = poselib.estimate_absolute_pose(
        image_points, object_points, camera, {"max_reproj_error": max_error}, {}
    )
    return pose, info


if __name__ == "__main__":
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Test a trained network on a specific scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "scene",
        type=Path,
        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"',
    )

    parser.add_argument(
        "network",
        type=Path,
        help="path to a network trained for the scene (just the head weights)",
    )

    parser.add_argument(
        "--encoder_path",
        type=Path,
        default=Path(__file__).parent / "ace_encoder_pretrained.pt",
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--set",
        default="test",
        help="file containing pre-trained encoder weights",
    )

    parser.add_argument(
        "--session",
        "-sid",
        default="",
        help="custom session name appended to output files, "
        "useful to separate different runs of a script",
    )

    parser.add_argument(
        "--image_resolution", type=int, default=480, help="base image resolution"
    )

    opt = parser.parse_args()

    device = torch.device("cuda")
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = opt.session

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,  # Default for ACE, we don't need scene coordinates/RGB-D.
        image_height=opt.image_resolution,
    )
    _logger.info(f"Test images found: {len(testset)}")

    # Setup dataloader. Batch size 1 by default.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # Load network weights.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    _logger.info(f"Loaded encoder from: {encoder_path}")
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")

    # Create regressor.
    network = Regressor.create_from_split_state_dict(
        encoder_state_dict, head_state_dict
    )

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Save the outputs in the same folder as the network being evaluated.
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f"test_{scene_name}_{opt.session}.txt"
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f"poses_{scene_name}_{opt.session}.txt"
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, "w", 1)
    pose_log = open(pose_log_file, "w", 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    pixel_grid_2HW = get_pixel_grid(network.OUTPUT_SUBSAMPLE)

    # Testing loop.
    testing_start_time = time.time()

    # xyz_map, image2points, image2name = read_nvm_file(opt.scene / "reconstruction.nvm")
    # name2id = {v: k for k, v in image2name.items()}
    count = 0

    with torch.no_grad():
        for (
            image_B1HW,
            image_ori,
            _,
            gt_pose_B44,
            gt_inv_pose_B44,
            _,
            intrinsics_B33,
            _,
            _,
            filenames,
            _,
            _,
            _,
        ) in tqdm(testset_loader):
            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast(enabled=True):
                features_BCHW = network.get_features(image_B1HW)
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently.
            for (
                frame_idx,
                (
                    scene_coordinates_3HW,
                    gt_pose_44,
                    gt_inv_pose_34,
                    intrinsics_33,
                    frame_path,
                ),
            ) in enumerate(
                zip(
                    scene_coordinates_B3HW,
                    gt_pose_B44,
                    gt_inv_pose_B44,
                    intrinsics_B33,
                    filenames,
                )
            ):
                # Extract focal length and principal point from the intrinsics matrix.
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                # We support a single focal length.
                assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

                xyz_arr = scene_coordinates_B3HW.view(3, -1).permute([1, 0]).numpy()
                uv_arr = (
                    pixel_grid_2HW[
                        :,
                        0 : scene_coordinates_B3HW.size(2),
                        0 : scene_coordinates_B3HW.size(3),
                    ]
                    .clone()
                    .view(2, -1)
                    .permute([1, 0])
                    .numpy()
                )
                # _, ind_pred = index.search(xyz_arr, 1)
                # mask = np.bitwise_not(np.isin(ind_pred[:, 0], bad_indices))
                # xyz_arr = xyz_arr[mask]
                # uv_arr = uv_arr[mask]

                pairs = []
                for j, (x, y) in enumerate(uv_arr):
                    xy = [x, y]
                    xyz = xyz_arr[j]
                    pairs.append((xy, xyz))
                pose, info = localize_pose_lib(pairs, focal_length, ppX, ppY)

                est_pose = np.vstack([pose.Rt, [0, 0, 0, 1]])
                est_pose = np.linalg.inv(est_pose)
                out_pose = torch.from_numpy(est_pose)

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                # Check various thresholds.
                if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                    pct1 += 1

                # Write estimated pose to pose file (inverse).
                out_pose = out_pose.inverse()

                # Translation.
                t = out_pose[0:3, 3]

                # Rotation to axis angle.
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle

                # Axis angle to quaternion.
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                # Write to output file. All in a single line.
                pose_log.write(
                    f"{frame_name} "
                    f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                    f"{t[0]} {t[1]} {t[2]} "
                    f"{r_err} {t_err} {0}\n"
                )

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    print(count / len(testset))

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info("Accuracy:")
    _logger.info(f"\t10cm/5deg: {pct10_5:.1f}%")
    _logger.info(f"\t5cm/5deg: {pct5:.1f}%")
    _logger.info(f"\t2cm/2deg: {pct2:.1f}%")
    _logger.info(f"\t1cm/1deg: {pct1:.1f}%")

    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")

    test_log.close()
    pose_log.close()