"""All pixel consumers use the same EXIF coordinate system."""

from io import BytesIO

import pytest
from PIL import Image, ImageOps

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import image_dimensions
from svsg.recognition.images import canonical_raster


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_pixel_coordinates_and_dimensions_agree(tmp_path, orientation):
    with Image.new("RGB", (12, 8), "red") as source:
        source.putpixel((0, 0), (0, 0, 255))
        exif = source.getexif()
        exif[274] = orientation
        stream = BytesIO()
        source.save(stream, format="PNG", exif=exif)
    data = stream.getvalue()
    path = tmp_path / "oriented.png"
    path.write_bytes(data)
    raster = RasterFunction.from_image(data)
    assert raster == RasterFunction.from_image(path) == canonical_raster(data)
    assert image_dimensions(data) == image_dimensions(path) == (raster.width, raster.height)
    with Image.open(BytesIO(data)) as raw, ImageOps.exif_transpose(raw) as oriented:
        assert raster.pixel(0, 0)[:3] == oriented.getpixel((0, 0))
