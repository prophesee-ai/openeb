import os
import numpy as np
import skimage.io

def pytestcase_tiff_images_read_write(tmpdir):
    # construct a random image with values in [0; 1]
    B, C, H, W = (1, 3, 8, 12)
    img_np = np.random.rand(B, C, H , W).astype(np.float16)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    assert img_np.min() == 0.0 and img_np.max() == 1.0

    # save the image as a tiff unit16
    img_tiff = (img_np.squeeze().transpose(1, 2, 0) * (2**16 - 1)).astype(np.uint16)
    assert img_tiff.shape == (H, W, C)
    filename = os.path.join(tmpdir, "test.tiff")
    skimage.io.imsave(filename, img_tiff)
    assert os.path.isfile(filename)

    # reload the tiff image and check that it is the same as the original
    img_tiff_reloaded = skimage.io.imread(filename)
    assert img_tiff_reloaded.shape == (H, W, C)
    assert (img_tiff_reloaded == img_tiff).all()
    img_np_reloaded = (img_tiff_reloaded.transpose(2, 0, 1).astype(np.float32) / (2**16 - 1))[None].astype(np.float16)
    assert img_np_reloaded.shape == (B, C, H, W)
    assert np.allclose(img_np_reloaded, img_np, atol=1e-4)