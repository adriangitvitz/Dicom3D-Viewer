# Dicom3D-Viewer

Usage:

```python
import dicomload

# Select the path to your dcm files
path = "<path-to-dicom-files>"
dicom_loader = dicomload.LoadDicoms(path)

# Get Planes based on the dcm images
planes = dicom_loader.load()

# Calculate pixel data
dicom_loader.calculate_hound_pixels()

# Get Surfaces and resize data
verts, faces = dicom_loader.mesh()

# 3D Plot in HTML format
dicom_loader.plot3d(verts, faces)
```


![Captura desde 2022-06-04 00-07-07](https://user-images.githubusercontent.com/39295224/171985342-d6ae8118-8364-414b-9761-5b96ab0624f0.png)
