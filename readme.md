[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
# MumfordShahCuda

This repository contains an implementation based on the paper:  
**Real-Time Minimization of the Piecewise Smooth Mumford-Shah Functional**, *E. Strekalovskiy, D. Cremers*, European Conference on Computer Vision (ECCV), 2014. ([pdf](https://vision.in.tum.de/_media/spezial/bib/strekalovskiy_cremers_eccv14.pdf)) ([video](https://vision.in.tum.de/_media/spezial/bib/strekalovskiy_cremers_eccv14.mp4))

![](https://vision.in.tum.de/_media/data/software/fastms.png)

## Generalization
The proposed variational model in the above paper is:  

min_u {sum_{x in Omega} |u(x)-f(x)|^2 + min(alpha*|nabla u|^2,lambda)}

The proposed **model has been adapted to an additional scalar operator a(x)** to solve a variational model of the form  

min_u {sum_{x in Omega} |a(x)u(x)-f(x)|^2 + min(alpha*|nabla u|^2,lambda)}  

## Features
- **GPU implementation** using CUDA, and a **CPU implementation**. Either implementation can be chosen at compile time using `CMake`.
- **Wrapper** for quick prototyping.

## Requirements

#### CUDA:

To use the **GPU implementation**, [CUDA](https://developer.nvidia.com/cuda-downloads) must be installed and *nvcc* must be in the current PATH.
The code is generated for NVIDIA GPUs of [compute capability](https://developer.nvidia.com/cuda-gpus).

*Note: You can still compile and use the* **_CPU version_**, *even if CUDA is not available. Just set `CUDAF=FALSE` in the `CMakeLists.txt` file*

#### OpenCV

[OpenCV](http://opencv.org/downloads.html) is only used for the executable (**NOT** Matlab/Mex) usages of the algorithm (*./main*). So you should have it installed in order to run the executable.

*Note: If OpenCV is not available and you do not want to run the executable but use only the Matlab version, then just set `EXEF=FALSE` in the `CMakeLists.txt` file*

#### MATLAB

If you want to use the provided **MATLAB wrapper**, MATLAB must be installed, and `MEXF=TRUE` in the `CMakeLists.txt` file as to be set.
Additionally, you have to set an environment variable `MATLAB_ROOT` pointing to the Matlab installation directory. In Ubuntu for Matlab R2018a type in Terminal or add the following line to your `.bashrc` file before running `CMake`:

  `export MATLAB_ROOT='/usr/local/MATLAB/matlab-R2018a'`

#### Notes
This code has been tested with the following setup  

- Ubuntu 16.04 with gcc 5.4  
- [optional] CUDA 8.0 and CUDA 9.1(CUDAF=FALSE must be set in CMakeLists.txt to build without CUDA support)  
- [optional] Matlab R2017b and R2018a (MEXF=FALSE must be set in CMakeLists.txt to build without Matlab/Mex support.)  
- [optional] OpenCV 2.4.9.1 (EXEF=FALSE must be set in CMakeLists.txt to build without executable and thus no OpenCV is needed.)

## Installation

Install:
In Terminal type

        git clone https://github.com/BjoernHaefner/MumfordShahCuda.git

        cd MumfordShahCuda

        mkdir build && cd build
		
		cmake ..
		
		make

## Usage
Run the code from command line (located in `build/bin`)
```
    ./mumfordShah -i <img.png> [-s <scalar_op.png>] [-m <mask.png>] [-o <output.png>]
	              [-p path/to/files/] [-l <float>] [-a <float>] [-iter <int>]
				  [-tol <float>] [-verbose <bool>] [-showall <bool>] [-show <bool>]
				  [-gamma <float>] [-sigma <float>] [-tau <float>]
```

## All Parameters

```
-i <string>
	(path to) image file}
	
-s <string> (optional)
	(path to) scalar_operator file.
	Default: identity

-m <string> (optional)
	(path to) mask file.
	Default: ones matrix
	
-o <string> (optional)
	output filename (will be stored in 'data').
	Default: no output is stored

-p <string> (optional)
	absolute path to data folder.
	Default: path/to/mumfordShah/data/
	
-l <float> (optional)
	trade-off parameter lambda.
	Default 0.1f
	
-a <float> (optional)
	parameter controlling the smoothness. alpha=infinity means piecewise constant result (can be triggered with alpha=-1).
	Default 10

-iter <int> (optional)
	number of iterations.
	Default 5000

-tol <float> (optional)
	Tolerance of stopping criterion.
	Default 1e-5f

-verbose <bool> (optional)
	Verbose output during algorithm.
	Default 1

-showall <bool> (optional)
	Show results on the fly after each iteration.
	Default 0

-show <bool> (optional)
	Show final result.
	Default 1

-gamma <float> (optional)
	Default 2

-sigma <float> (optional)
	Initial step size of dual variable
	Default 0.5f

-tau <float> (optional)
	Initial step size of primal variable
	Default 0.25f
```

## Examples
We provide examples for running the executable as well as how to run the code in matlab. Have a look in `MumfordShahCuda/examples/`  

- For the executable run in Terminal  

        bash MumfordShahCuda/examples/example.sh
		
- For the Matlab code run in Matlab

        MumfordShahCuda/examples/example.m

- If you want to run the algorithm from command line, you can for example run

        ./mumfordShah -i intensity.png -m mask.png -s shading.png -o result.png -a -1

## Tips and troubleshooting
* If Matlab throws an error like 

    ```Invalid MEX-file '/mumfordshah/build/lib/mumfordShahMEX.mexa64':
    /MATLAB/R2016b/bin/glnxa64/../../sys/os/glnxa64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /mumfordshah/build/lib/mumfordShahMEX.mexa64)```

    then start Matlab from terminal and preload the `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` library:

    ```LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6" matlab```

* If Matlab throws an error like

    ```Invalid MEX-file '/usr/wiss/haefner/Documents/coding/cpp/mumfordshah/build/lib/mumfordShahMEX.mexa64':
    /usr/wiss/haefner/Documents/coding/cpp/mumfordshah/build/lib/mumfordShahMEX.mexa64: undefined symbol: _ZN2cv8fastFreeEPv```

    then build the mex file with the `EXEF=FALSE` option.

* If Matlab keeps crashing: Check if you use `clear all` somewhere in your Matlab code. Exchange it with `clear`.


## Citation
If you make use of the library in any form in a scientific publication, please refer to `https://github.com/BjoernHaefner/MumfordShahCuda` and cite following papers

```
@inproceedings{strekalovskiy2014realtime,
 author = {E. Strekalovskiy and D. Cremers},
 title = {{Real-Time Minimization of the Piecewise Smooth Mumford-Shah Functional}},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2014},
 pages = {127-141},
 keywords = {convex-relaxation},
}
```
```
@inproceedings{haefner2018fight,
 title = {{Fight ill-posedness with ill-posedness: Single-shot variational depth super-resolution from shading}},
 author = {B. Haefner and Y. Quéau and T. Möllenhoff and D. Cremers},
 booktitle = {I{EEE}/{CVF} {C}onference on {C}omputer {V}ision and {P}attern {R}ecognition (CVPR)},
 year = {2018},
 keywords = {rgb-d,reconstruction,3d-reconstruction,photometry,variational,super-resolution,photometricdepthsr},
}
```
