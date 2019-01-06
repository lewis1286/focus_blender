# Focus Blender

For macro photography.  Take a bunch of images of the same small object, only
varying the focus area.  Take enough so that all parts of the subject are in
focus in at least one image.

Edit the end of the python file to point to the directory of the images.

It will merge them into one image, taking the in-focus parts of each input
image.

This is done by roughly the following steps:

1. align images (this may cause more noise than is worth)
    FIXME: add docopt and make align optional

2. Take derivative of all images, and threshold for high absolute values.  The
   idea is that the image is sharper where in focus, so we capture areas of
   focus in this way.  These will be come masks.

3.  Smooth and normalize the thresholded masks into weight matrices same dims
    as the input images.

4.  Take a linear combination of images multiplied by the weight matrices as
    the output.

## Usage:

Edit the path at the bottom of `focus_blender.py` to path of the images to
merge

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python focus_blender.py
```
There will now be a merged image in the directory of the input images.

