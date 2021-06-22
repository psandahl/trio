import argparse

import trio.pose_estimate as tp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3d playground")
    parser.add_argument("-f", "--file", type=str, help="comparison json file")
    parser.add_argument("-c", "--confidence", type=float,
                        default=0.7, help="filter")
    parser.add_argument("-e", "--error", action="store_true",
                        help="simulate fov error")

    args = parser.parse_args()

    if not args.file is None:
        tp.run(args.file, args.confidence, args.error)
    else:
        print("A file must be given")
