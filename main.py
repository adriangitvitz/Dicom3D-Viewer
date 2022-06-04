#!/usr/bin/env python3
import dicomload

path = "<path-to-dicom-files>"
dicom_loader = dicomload.LoadDicoms(path)
planes = dicom_loader.load()
dicom_loader.calculate_hound_pixels()
verts, faces = dicom_loader.mesh()
dicom_loader.plot3d(verts, faces)
