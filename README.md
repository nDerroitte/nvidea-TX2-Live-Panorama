# Installation and usage of the code
## About
The code was designed in two parts. 
The goal of the first part was to make a panorama from a video stream. This last can be inputed or taken live.
In the second part, we added a motion and person detection module.

To run the last version of the project, one should thus use the Part2 folder of this repository.
## Installation
The first step is to install the following package on the Jetson : 
```sh
$ pip3 install imutils
```
After this installation, one the place  all the code in a folder and openning a terminal on the according folder and there is nothing much to do as far as installation is concerned.
Let's just note that the Jetson TX2 should be installed as described in the statement and therefore, OpenCV2, numpy, ... should be already installed.

Some part of the code may requiere to boost the performance of the TX2 to run properly/quicker.
Enabling Denver2 and increasing CPU/GPU frequency (See [Jetson Hack page on the subject](https://www.jetsonhacks.com/2017/03/25/nvpmodel-nvidia-jetson-tx2-development-kit/)): 
```sh
$ sudo nvpmodel -m0
```
Maximize clock governor frequencies:
```sh
$ sudo /home/nvidia/jetson_clocks.sh
```

The need to apply these lines will be commented for each of the different parts.
## Usage
### Image/Video acquisition
The first step is to perform the Image/Video acquisition. In order to do so, one can simply run the following line :
```sh
$ python3 CameraCapture.py LABEL REF
```
where  '[LABEL] [REF]'' are 2 inputs to specify as :
* Label : int number : 1 for indoor and 2 for outdoor
* Ref :  boolean : True if the sequence is a reference one (without moving object) and False otherwise

Once the window is opened, the user can use the following key to perform the corresponding actions : 
* `p` : Capture a single image in Capture/current\_date 
* `d` : Record a 500 frames sequence in Capture/current\_date 
* `v` : Record a 1500 frames sequence in Capture/current\_date 
* `q` : Close the windows

As explain in the report, it is mandatory to boost the performance of the Jetson TX2 for this code. Ohterwise, it won't capture excalty 25 frames per second.
### Camera calibration
The camera calibration can be done using a chessboard print on a flat surface of A4 paper size using the following line of code :
```sh
python3 CameraCalibration.py
```
In order to start the calibration, the user can simply press 't'. Once the  calibration has started, the algorithm will try to identify a pattern on what it's filming. Once done, the user can accept the image or not. One must take a average of 30 images to perform a good camera calibration and repress 't' for each of them.
The output Json file in registred in : Capture/current_time.json

The Jetson TX2 boosting can be use in order to increase the performance of the algorithm but is not mandatory.
### Creating a panoramic image from a folder of jpg image
The command in charge of the creation of the panoramic image contained a few arguments that are detailed here : 

```sh
python3 main.py CAMERA_MATRIX.json {panorama, matching_demo} [FOLDER PATH]
```
* `CAMERA_MATRIX.json` : Camera matrix computed with the camera calibration, store in a Json file. CAMERA_MATRIX should be change with the name/path of the Json object. (Using our CameraCalibration.py script, the Json of the camera matrix is store in Capture/time_of_the_calibration.json )
* `{panorama, matching_demo}`: Specify one of the possible arguments:
  * ``panorama`` should be use to create a panorama. The output panorama register itself 
  * `matching_demo` should be use to see how the matching of feature take place. Does not create any results and only help for better understanding option. The following inputs can be used in this case :
    * `p`: Pause/Unpause the stream.
    * `r`: Restart display.
    * `q`: quit
* `FOLDER_PATH`: Path of the folder containing the sequence to consider. If the argument is not filled, the algorithm runs in live mode.
  * While using the panorama feature in live mode, the user should press `s`to start the panorama and repress `s`. It can also press `q` to quit at anytime.
  * While using the matching_demo feature in live, the user input does not change from the non-live mode.

Note that only the 2 first arguments are mandatory. Not giving a 3rd argument puts the algortihm in live mode. 

The Jetson TX2 boosting can be use in order to increase the performance of the algorithm but is not mandatory.

### Motion detection 

In order to access the motion detection, one can use the following code : 
```sh
python3 main.py CAMERA_MATRIX.json motion_detection [FOLDER PATH, PERFORMANCE ASSESSMENT]
```
* `CAMERA_MATRIX.json` : Camera matrix computed with the camera calibration, store in a Json file. CAMERA_MATRIX should be change with the name/path of the Json object. (Using our CameraCalibration.py script, the Json of the camera matrix is store in Capture/time_of_the_calibration.json )
* `FOLDER_PATH` : Path of the folder containing the sequence to consider. If the argument is not filled, the algorithm runs in live mode.
* `PERFORCEMENT ASSESSMENT`: If specify (with the value True), enable the perforcement assessment of the motion detection module. Will print the mean error of all erros made by the module (See report for detail about these errors). An  **Annotation/{In, Out}/** folder containing the reference masks and the box text file should be present in the current folder.
If not specify : the output is the sequence where the object in motion are framed with a red rectangles.

Note that if run in live (not filling the `FOLDER PATH` argument), the code can not be used to performe assessment. Nevertheless, if run on a pre-defined sequence (by filing `FOLDER PATH`), the `PERFORCEMENT ASSESSMENT`parameter can be set to True or False. As explained previously, the output will depend on this parameter.

If run in live the user can use the following inputs : 
* `s` : start the motion detection
* `q`: quit

If run from existing sequences : 
* `p`: Pause/Unpause the stream.
* `r`: Restart display.
* `q`: quit

The Jetson TX2 boosting can be use in order to increase the performance of the algorithm but is not mandatory.
### Person detection
To run the person detection module, one should use the following line : 
```sh
python3 main.py CAMERA_MATRIX.json person_detection [FOLDER PATH, PERFORMANCE ASSESSMENT]
```
The arguments work exactly as for the Motion detection module : 
* `CAMERA_MATRIX.json` : Camera matrix computed with the camera calibration, store in a Json file. CAMERA_MATRIX should be change with the name/path of the Json object. (Using our CameraCalibration.py script, the Json of the camera matrix is store in Capture/time_of_the_calibration.json )
* `FOLDER_PATH` : Path of the folder containing the sequence to consider. If the argument is not filled, the algorithm runs in live mode.
* `PERFORCEMENT ASSESSMENT`: If specify (with the value True), enable the perforcement assessment of the person detection module. Will print the mean error of all erros made by the module (See report for detail about these errors). An  **Annotation/{In, Out}/** folder containing the reference masks and the box text file should be present in the current folder.
If not specify : the output is the sequence where the humain are framed with a red rectangles.

Note that if run in live (not filling the `FOLDER PATH` argument), the code can not be used to performe assessment. Nevertheless, if run on a pre-defined sequence (by filing `FOLDER PATH`), the `PERFORCEMENT ASSESSMENT`parameter can be set to True or False. As explained previously, the output will depend on this parameter.

If run in live the user can use the following inputs : 
* `s`: start the motion detection
* `q`: quit

If run from existing sequences : 
* `p`: Pause/Unpause the stream.
* `r`: Restart display.
* `q`: quit

The Jetson TX2 boosting can be use in order to increase the performance of the algorithm but is not mandatory.
### Enhanced panoramic image
To create the enhanced panoramic image, where the moving objects have been removed, the following command should be use : 
```sh
python3 main.py CAMERA_MATRIX.json enhanced_panorama [FOLDER PATH]
```
The output panorama registers itself in **./enhanced_panorama.jgp**. 
* `CAMERA_MATRIX.json` : Camera matrix computed with the camera calibration, store in a Json file. CAMERA_MATRIX should be change with the name/path of the Json object. (Using our CameraCalibration.py script, the Json of the camera matrix is store in Capture/time_of_the_calibration.json )
* `FOLDER PATH` : Path of the folder containing the sequence to consider. If the argument is not filled, the algorithm runs in live mode.

If run in live the user can use the following inputs : 
* `s`: start the motion detection
* `q`: quit

If run from existing sequences : 
* `p`: Pause/Unpause the stream.
* `r`: Restart display.
* `q`: quit

The Jetson TX2 boosting can be use in order to increase the performance of the algorithm but is not mandatory.
