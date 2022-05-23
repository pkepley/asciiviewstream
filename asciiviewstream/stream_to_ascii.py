import os
import argparse
from math import floor, exp, log
import cv2
import numpy as np
import colored.hex as chex
from scipy.spatial import KDTree

# clear
_clear_cmd_ = "cls" if os.name == "nt" else "clear"

# some density variables, hackily used as globals here (:
density = "N@#W$9876543210?!abc;:+=-,._ "
n_lvl = len(density)
X = exp(log(255) / n_lvl)
log_X = log(X)

# some color variables, hackily provided as globals (:
color_map = {v: k for k, v in chex._xterm_colors.items()}
color_list = list(color_map.keys())
color_list_255 = 255 * np.array([
    list(map(lambda x: int(x, 16), [c[1:3], c[3:5], c[5:7]]))
    for c in color_list
], dtype="uint8")
kdd = KDTree(color_list_255)


def rescale_shape(shape, rescale_x=None, rescale_y=None, font_aspect=2.0):
    """
    rescale an image
    """

    h, w = shape[0], shape[1]

    if rescale_x is not None and rescale_y is not None:
        rescale = (floor(font_aspect * rescale_x), rescale_y)

    elif rescale_y is not None:
        rescale = (floor(font_aspect * (w/h) * rescale_y), rescale_y)

    elif rescale_x is not None:
        rescale = (floor(rescale_x), floor((h/w) * rescale_x / font_aspect))

    return rescale


def best_color(rgb):
    """
    Find the closest color in the color palette
    """
    bc = np.array([
        color_map[color_list[i]]
        for i in kdd.query(rgb.reshape((-1, 3)))[1]
    ])
    return bc.reshape(rgb.shape[0:2])


def img_to_ascii(im,
                 rescale,
                 log_scale=True,
                 invert_intensity=False,
                 colorize=False):
    """
    Convert img to ascii string (optionally colorize)
    """
    im = cv2.resize(im, rescale)
    im_s = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if invert_intensity:
        im_s = 255 - im_s

    # compute the nearest neighbor color in the 256 color pallete
    if colorize:
        im_c = best_color(255 - cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # encode the intensity
    if log_scale:
        lvl = np.minimum(np.floor(np.log(im_s) / log_X), n_lvl - 1)
        lvl = np.maximum(lvl, 0)
    else:
        lvl = np.minimum(np.floor(n_lvl * (im_s / 255.0)), n_lvl - 1)
    lvl = lvl.astype("int")

    # construct the ascii string
    ny, nx = im_s.shape
    ascii = ""
    for i in range(ny):
        for j in range(nx):
            # using xterms 256 colors...
            if colorize:
                # This was TOO SLOW:
                # ascii += stylize(density[lvl[i, j]], fg(im_c[i, j]))

                # This works well enough:
                ascii += (
                    f"\x1b[38;5;{str(im_c[i,j])}m" +
                    f"{density[lvl[i, j]]}\x1b[m"
                )
            else:
                ascii += density[lvl[i, j]]
        ascii += "\n"

    return ascii


def main(rescale_x=None,
         rescale_y=None,
         font_aspect=2.0,
         log_scale=False,
         invert_intensity=False,
         colorize=False,
         show_camera=False,
         frame_refresh_rate=100):
    """
    run the img --> ascii program against a live-stream
    """

    # open the input camera feed
    vc = cv2.VideoCapture(-1)
    if vc.isOpened():
        rval, frame = vc.read()
        rescale = rescale_shape(frame.shape, rescale_x, rescale_y, font_aspect)
    else:
        rval = False

    # open the output camera feed
    if show_camera:
        cv2.namedWindow("camera_stream")

    # print output
    print_buf = ""

    # main loop
    while rval:
        im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if show_camera:
            cv2.imshow("camera_stream", im_gray)

        print_buf = img_to_ascii(
            frame,
            rescale=rescale,
            log_scale=log_scale,
            invert_intensity=invert_intensity,
            colorize=colorize
        )
        os.system(_clear_cmd_)
        print(print_buf)

        rval, frame = vc.read()
        key = cv2.waitKey(frame_refresh_rate)
        if key == 27:
            break

    # clean up
    if show_camera:
        cv2.destroyWindow("camera_stream")
    vc.release()


if __name__ == "__main__":
    # get command line arguments
    parser = argparse.ArgumentParser(
        description="Stream video to ASCII terminal characters"
    )
    parser.add_argument(
        "-x", metavar="rescale_x", type=int,
        help="number of character columns (default is 100)"
    )
    parser.add_argument(
        "-y", metavar="rescale_y", type=int,
        help="number of character rows " +
             "(default computed based on rescale_x and font_aspect)"
    )
    parser.add_argument(
        "-f", metavar="frame_refresh", type=int, default=100,
        help="time between frame refreshes in ms (default is 100)"
    )
    parser.add_argument(
        "-a", metavar="font_aspect", type=float, default=2.0,
        help="font aspect ratio (default is 2.0)")
    parser.add_argument(
        "-l", action="store_true",
        help="apply linear intensity scaling (default is log scaling)"
    )
    parser.add_argument(
        "-i", action="store_true", help="invert intensity"
    )
    parser.add_argument(
        "-c", action="store_true", help="colorize console output"
    )
    parser.add_argument(
        "-s", action="store_true", help="show video stream"
    )
    args = parser.parse_args()

    # default to 150 px wide if not provided
    if args.x is None and args.y is None:
        args.x = 100

    # run program
    main(rescale_x=args.x,
         rescale_y=args.y,
         font_aspect=args.a,
         frame_refresh_rate=args.f,
         log_scale=not args.l,
         invert_intensity=args.i,
         colorize=args.c,
         show_camera=args.s)
