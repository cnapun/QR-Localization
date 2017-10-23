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

## Notes/Obeservations
* I could not determine the optimal camera matrix, as using `cv2.calibrateCamera` would output very inconsistent camera matrices. One thing to note is that the rotation vectors from using `calibrateCamera` seem to better reflect the yaw (and maybe roll) of the patterns, but the translation vectors and pitch do not, so I could use `calibrateCamera` only for those two components of rotation, and `solvePnP` for translation. I ended up using camera matrix values from a website I found, so they might not be very accurate.
* My method of determining the QR code corners in the image will not scale to more complex patterns (due to the call of `itertools.permutations`). I used this because I found that the orientation of QR codes can be found from only using the 3 corners, but I couldn't capture the 3 corners without the fourth as well.
* My code assumes the 8.8 cm measurement is for the size of the pattern, not the size of the entire `pattern.png`
* I tried a simple SIFT knn matching, but it didn't seem to work as well
