# CORPCA-OF
**Compressive Online Robust Principal Component Analysis with Optical Flow (CORPCA-OF)**

    Version 1.1,  May 1, 2018
    Implementations by Srivatsa Prativadibhayankaram (developer) and Huynh Van Luong (adviser), 
    Email: srivatsa.pv@live.com and huynh.luong@fau.de,
    Multimedia Communications and Signal Processing, University of Erlangen-Nuremberg.  
  
  `Please see` [LICENSE](https://github.com/huynhlvd/corpca-of/blob/master/LICENSE.md) `for the full text of the license.`

Please cite this publication: 

`Srivatsa Prativadibhayankaram, Huynh Van Luong, Thanh Ha Le, and André Kaup, "`
[Compressive Online Robust Principal Component Analysis with Optical Flow for Video Foreground-Background Separation](https://doi.org/10.1145/3155133.3155184)'," Proceedings of the Eighth International Symposium on Information and Communication Technology. ACM, 2017, pp.385–392.     

**_Solving the problem and Updating priors using Optical Flow_**

<img src="https://latex.codecogs.com/svg.latex?\small&space;\min_{\boldsymbol{x}_{t},\boldsymbol{v}_{t}}\Big\{H(\boldsymbol{x}_{t},\boldsymbol{v}_{t})=\frac{1}{2}\|\mathbf{\Phi}(\boldsymbol{x}_{t}&plus;\boldsymbol{v}_{t})-\boldsymbol{y}_{t}\|^{2}_{2}&space;&plus;\lambda\mu\sum\limits_{j=0}^{J}\beta_{j}\|\mathbf{W}_{j}(\boldsymbol{x}_{t}-\boldsymbol{z}_{j})\|_{1}&plus;&space;\mu\Big\|[\boldsymbol{B}_{t-1}\boldsymbol{v}_{t}]\Big\|_{*}\Big\}" title="\small \min_{\boldsymbol{x}_{t},\boldsymbol{v}_{t}}\Big\{H(\boldsymbol{x}_{t},\boldsymbol{v}_{t})=\frac{1}{2}\|\mathbf{\Phi}(\boldsymbol{x}_{t}+\boldsymbol{v}_{t})-\boldsymbol{y}_{t}\|^{2}_{2} +\lambda\mu\sum\limits_{j=0}^{J}\beta_{j}\|\mathbf{W}_{j}(\boldsymbol{x}_{t}-\boldsymbol{z}_{j})\|_{1}+ \mu\Big\|[\boldsymbol{B}_{t-1}\boldsymbol{v}_{t}]\Big\|_{*}\Big\}" />

Inputs:
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\boldsymbol{y}_{t}\in&space;\mathbb{R}^{m}" title="\boldsymbol{y}_{t}\in \mathbb{R}^{m}" />: A vector of observations/data <br /> 
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\mathbf{\Phi}\in&space;\mathbb{R}^{m\times&space;n}" title="\mathbf{\Phi}\in \mathbb{R}^{m\times n}" />: A measurement matrix <br />
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\boldsymbol{Z}_{t-1}:=\{\boldsymbol{z}_{j}\}_{j=0}^J\in&space;\mathbb{R}^{n}" title="\boldsymbol{y}_{t}\in \mathbb{R}^{n}" />: The foreground prior <br />
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\boldsymbol{B}_{t-1}\in&space;\mathbb{R}^{n\times&space;d}" title="\boldsymbol{B}_{t-1}\in \mathbb{R}^{n\times d}" />: A matrix of the background prior, which could be initialized by previous backgrounds <br />

Outputs:
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\boldsymbol{x}_{t},\boldsymbol{v}_{t}\in\mathbb{R}^{n}" title="\boldsymbol{x}_{t},\boldsymbol{v}_{t}\in\mathbb{R}^{n}" />: Estimates of foreground and background
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\boldsymbol{Z}_{t}:=\{\boldsymbol{z}_{j}=\boldsymbol{x}_{t-J&plus;j}\}" title="\boldsymbol{Z}_{t}:=\{\boldsymbol{z}_{j}=\boldsymbol{x}_{t-J+j}\}" />: The updated foreground prior
- <img src="https://latex.codecogs.com/svg.latex?\dpi{150}&space;\boldsymbol{B}_{t}\in&space;\mathbb{R}^{n\times&space;d}" title="\boldsymbol{B}_{t}\in \mathbb{R}^{n\times d}" />: The updated background prior

**_How to run:_**

1. Open CMake GUI
2. Drag and drop CmakeLists.txt from the "CORPCA-OF" folder into CMake
3. Select output path, type in "build" after the path
4. Select configure, choose your compiler (VS 2017/2015 etc.)
5. After no errors are present, click generate
6. If you go to the CORPCA-OF folder, you will see a build folder created, open it
7. Open the visual studio project that is created
8. Build the solution
9. Select the project demoCORPCA-OF and set it as start up project
10. Now select "Run"
11. Enter path to the video files (ex D:\inData)
12. Enter path where output has to be saved (ex D:\outDataOF)
13. Enter sequence name (Bootstrap, Curtain etc.)
14. Enter the format name (bmp, png etc.)
14. Enter scaling factor (1.0, 0.5, 0.25)
15. Enter rate (1.0, 0.8, 0.6, 0.4, 0.2)
16. Check the output folder that was specified to see the seperated images

NOTE: 
* Inside the output folder, please create folders with names of video sequences, i.e., for every sequence in inData, create a new empty folder inside the output path folder (D:\outDataOF)
* Create a folder named prior inside video sequence folder (ex in inData)
* For every sequence in inData, create a new folder inside the prior folder
Here is a sample folder structure:

- inData
    * Bootstrap 
    * Curtain 
    * ...
    * prior 
      + Bootstrap (to be created)
      + Curtain (to be created)
      + ...
- outDataOF
    * Bootstrap (to be created)
    * Curtain (to be created)
    * ...

