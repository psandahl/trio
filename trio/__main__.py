import argparse
import open3d

import trio.pose_estimate as tp
import trio.epipolar_check as te
import trio.pointcloud as tpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3d playground")
    parser.add_argument("-m", "--mode", type=str,
                        default="cons", choices=["pose", "cons", "view", "cloud"],
                        help="Mode")
    parser.add_argument("-f", "--file", type=str, help="comparison json file")
    parser.add_argument("-p", "--ply", type=str, help="ply file")
    parser.add_argument("-v", "--video", type=str, help="video file")
    parser.add_argument("--epithres", type=float,
                        help="Epipolar treshold", default=0.5)
    parser.add_argument("--pointdist", type=float,
                        help="Point distance", default=0.5)
    parser.add_argument("--bufferwidth", type=int,
                        help="Buffer width", default=30)
    parser.add_argument("-c", "--confidence", type=float,
                        default=0.7, help="filter")
    parser.add_argument("-w", "--width", type=int,
                        default=1280, help="Image width")
    parser.add_argument("-g", "--height", type=int,
                        default=720, help="Image height")
    parser.add_argument("-e", "--error", action="store_true",
                        help="simulate fov error")

    args = parser.parse_args()

    if args.mode == "pose" and not args.file is None:
        tp.run(args.file, args.confidence, args.error)
    elif args.mode == "cons" and args.file is None:
        te.run_stereo_normal()
    elif args.mode == "cons" and not args.file is None:
        te.run(args.file, args.width, args.height)
    elif args.mode == "cloud" and not args.video is None and not args.ply is None \
            and not args.file is None:
        tpc.run(args.video, args.file, args.ply, point_dist=args.pointdist,
                buffer_width=args.bufferwidth, epi_thres=args.epithres)
    elif args.mode == "view" and not args.ply is None:
        pcd = open3d.io.read_point_cloud(args.ply)
        open3d.visualization.draw_geometries([pcd])
    else:
        print(parser.format_help())
