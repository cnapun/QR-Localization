# QR-Localization
My solution to the DroneDeploy coding challenge.

## Requirements
Numpy, OpenCV (cv2)
Tested with Numpy 1.13.3, OpenCV 3.3.0, and Python 3.6.3

## Solution
Sample output can be found in `sample_output.txt`. To see the simulated pictures, please view `web/vis.html` (using a server). If using Python, run `python3 -m http.server` from the root project directory, and then navigate to http://localhost:8000/web/vis.html in a web browser (Chrome tested). To switch between images, select from the dropdown. There is not currently an easy way to visualize new images; it would be simplest to simply change the default parameters in `vis.html`.

To run the calculations, use `python3 find_params.py -i PATH_TO_IMAGE -p PATH_TO_PATTERN`. By default, the pattern is assumed to be at `Camera Localization/pattern.png`. This will produce output of the form
```
Camera Angles:
Roll: 43.2°, Pitch: 10.1°, Yaw: 29.9°
Camera Position:
(x,y,z) = (-159, 48, 891) mm
```

