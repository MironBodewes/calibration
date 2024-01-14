### how to use:
1. start 1_make_images.py and make some images (will be in `<workingdirectory>/data/`)  
you could also just make the images yourself with another program, but there is a naming convention.
1. start 2_undistort_and_measure.   
this will compute the camera matrix and distortion matrix with the intrinsic and extrinsic values from the images you made.
1. start Distanzmessung.ipynb
1. result will be shown at the end of the notebook







### some thoughts
aktuell: pixel_per_cm als Skalar

eigentlich besser: pixel_per_cm als 2D-Array und zusätlich muss die Orientierung noch beachtet werden, weil der Stift, wenn er von der Kamera wegzeigt, wahrscheinlich weniger Pixel füllt als wenn er Waagerecht im Bild liegt

