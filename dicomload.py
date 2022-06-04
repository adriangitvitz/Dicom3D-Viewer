from typing import Any, List, Tuple
import numpy as np
import pydicom
import os
from pydicom.dataset import FileDataset
from pydicom.dicomdir import DicomDir
from plotly import figure_factory as FF
import matplotlib.pyplot as plt
import scipy.ndimage
from plotly.offline import plot
from skimage import measure

class LoadDicoms:
    def __init__(self, path) -> None:
        self.path = path
        self.hound_pixels = np.array([])
        self.planes = []

    def load(self):
        planes: list[FileDataset | DicomDir] = [pydicom.read_file("{}/{}".format(self.path, s)) for s in os.listdir(self.path)]
        print("Loading {} DICOM files...".format(len(planes)))
        planes.sort(key=lambda x: int(x.InstanceNumber))
        try:
            w_planes = np.abs(planes[0].ImagePositionPatient[2] - planes[1].ImagePositionPatient[2])
        except:
            w_planes = np.abs(planes[0].SliceLocation - planes[1].SliceLocation)
        for p in planes:
            p.SliceThickness = w_planes
        self.planes = planes

    def calculate_hound_pixels(self):
        print("Calculating pixel data...")
        image = np.stack([
            plane.pixel_array for plane in self.planes
        ])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = self.planes[0].RescaleIntercept
        slope = self.planes[0].RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        self.hound_pixels = np.array(image, dtype=np.float64)

    def show_samples(self, rows=3, cols=3, start_with=1, interval=4, show=False):
        # TODO: Improve Sampling
        _, ax = plt.subplots(rows,cols,figsize=[50,50])
        print(len(self.hound_pixels))
        for i in range(len(self.hound_pixels)):
            index = start_with + i*interval
            ax[int(i/rows), int(i%rows)].set_title('Sample {}'.format(i))
            ax[int(i/rows), int(i%rows)].imshow(self.hound_pixels[i],cmap='gray')
            ax[int(i/rows), int(i%rows)].axis('off')
        if show:
            plt.show()
        else:
            print("Saving samples in PNG format...")
            plt.savefig("dicom_samples.png")

    def modify_samples(self, space=[1,1,1]):
        # TODO: Improve Interpolation
        print("Resizing data...")
        map_thickness = map(float, ([self.planes[0].SliceThickness]))
        pixel_spacing = map(float, (self.planes[0].PixelSpacing))
        map_thickness = map(sum, zip(map_thickness, pixel_spacing))
        map_thickness = np.array(list(map_thickness))
        space_resize  = map_thickness / space
        temp_shape    = self.hound_pixels.shape * space_resize
        n_shape       = np.round(temp_shape)
        n_resize      = n_shape / self.hound_pixels.shape
        new_image     = scipy.ndimage.interpolation.zoom(self.hound_pixels, n_resize)
        return new_image

    def mesh(self, step_size=1):
        print("Creating Mesh...")
        image = self.modify_samples()
        surface_transpose = image.transpose(2,1,0)
        data: Tuple[Any, ...] = measure.marching_cubes(surface_transpose, step_size=step_size, method='lewiner', allow_degenerate=True)
        verts: List[List[Any]]
        faces: List[List[Any]]
        verts, faces, _, _ = data
        return verts, faces

    def plot3d(self, verts, faces):
        print("Creating 3D Mesh...")
        x,y,z = zip(*verts)
        colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
        fig = FF.create_trisurf(x=x,y=y,z=z,plot_edges=False,colormap=colormap,simplices=faces,backgroundcolor='rgb(64,64,64)',title="Visualizacion en 3D")
        plot(fig)
