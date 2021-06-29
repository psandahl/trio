import argparse

import trio.pose_estimate as tp
import trio.epipolar_check as te

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3d playground")
    parser.add_argument("-m", "--mode", type=str,
                        default="cons", choices=["pose", "cons"], help="Mode")
    parser.add_argument("-f", "--file", type=str, help="comparison json file")
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
    else:
        print(parser.format_help())
