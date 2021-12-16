![SUPRA Logo](http://campar.in.tum.de/files/goeblr/supra_logo_full_small.png "SUPRA Logo")

Notes:
* This branch already merged the https://github.com/intel/supra-on-oneapi/ with https://github.com/IFL-CAMP/supra.
* Please follow https://github.com/intel/supra-on-oneapi/#readme to run the test.


SUPRA: Open Source Software Defined Ultrasound Processing for Real-Time Applications
================

By the [Chair for Computer Aided Medical Procedures](http://campar.in.tum.de/)

[![TUM](http://campar.in.tum.de/files/goeblr/TUM_Web_Logo_blau.png "TUM Logo")](http://tum.de)

Main contributors: 

* R&uuml;diger G&ouml;bl
* Dr. Christoph Hennersperger

Supported by [EDEN2020](http://eden2020.eu)

[![EDEN2020 Logo](http://campar.in.tum.de/files/goeblr/EDEN2020_Logo_Small.jpg "EDEN2020 Logo")](http://eden2020.eu)


A 2D and 3D Pipeline from Beamforming to B-mode
----------------

**SUPRA** is an open-source pipeline for fully software 
defined ultrasound processing for real-time applications.
Covering everything from beamforming to output of B-Mode images, SUPRA
can help reproducibility of results and allows modifications to the image acquisition.

Including all processing stages of a usual ultrasound pipeline, it can be executed in 2D and 3D on consumer GPUs in real-
time. Even on hardware as small as the CUDA enabled Jetson TX2 **SUPRA** can be run for 2D imaging in real-time.

![Standard ultrasound pipeline and where the processing takes place. Transmit beamforming is performed on the CPU, transmit and receive are performed in specialized hardware. All other processing steps (receive beamforming, envelope detection, log-compression, scan-conversion) happen in software and on the GPU](http://campar.in.tum.de/files/goeblr/UsPipeline_small.png "Standard pipeline and where the processing takes place")

Getting started
----------------
To get an overview of the concepts behind SUPRA and how you can use and modify it, head over to the [wiki](https://github.com/IFL-CAMP/supra/wiki), or have a look at the recording of SUPRA-con at YouTube [SUPRA-con Playlist](https://www.youtube.com/watch?v=dQkUSpV2CtE&list=PLWMNh90FOFSiGOA_VilmuRVqD_5goBrJq).

License
----------------
LGPL v2.1
see [LICENSE](LICENSE)

Publication
----------------
If you use SUPRA for your research, please cite our work
[https://doi.org/10.1007/s11548-018-1750-6](https://doi.org/10.1007/s11548-018-1750-6)

G&ouml;bl, R., Navab, N. & Hennersperger, C. , "SUPRA: Open Source Software Defined Ultrasound Processing for Real-Time Applications" Int J CARS (2018). https://doi.org/10.1007/s11548-018-1750-6

	@Article{Goebl2018supra,
		author="G{\"o}bl, R{\"u}diger and Navab, Nassir and Hennersperger, Christoph",
		title="SUPRA: open-source software-defined ultrasound processing for real-time applications",
		journal="International Journal of Computer Assisted Radiology and Surgery",
		year="2018",
		month="Mar",
		day="28",
		issn="1861-6429",
		doi="10.1007/s11548-018-1750-6",
		url="https://doi.org/10.1007/s11548-018-1750-6"
	}

Building
----------------
### Requirements

* cmake &ge; 3.4
* gcc &ge; 4.8 or min. Visual Studio 2015 (Compiler needs to be supported by CUDA! For that, see the CUDA installation instructions.)
* QT &ge; 5.5
* TBB
* CUDA &ge; 10.0

	
### Build instructions (Ubuntu 16.04 / 18.04)

Install CUDA (&ge; 10.0) as described by NVIDIA https://developer.nvidia.com/cuda-downloads .
Keep in mind that the C++ host compiler has to be supported by the CUDA version.
(Check http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html and http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html for details.)

Build requirements

	apt-get install cmake cmake-gui qt5-default libtbb-dev libopenigtlink-dev git
	
SUPRA

	mkdir -p $HOME/git && cd $HOME/git #(or your favorite directory for repositories)
	git clone https://github.com/IFL-CAMP/supra.git
	cd supra
	mkdir -p build && cd build
	cmake-gui ..
	
1. Configure
2. For systems with multiple gcc versions, make sure to select one supported by the installed CUDA version
3. You might need to specify the CUDA toolkit directory (usually "`/usr/local/cuda`")
4. Configure & Generate, then close cmake and build
5. Build SUPRA
	
	make -j5
	
6. Start SUPRA: See below

### Building with PyTorch inference (via libtorch)

1. Download the stable libtorch for CUDA 10.0 from pytorch.org
   
        https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-latest.zip
        https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-latest.zip 
   
   Tested version: 1.1.0:
   
        https://download.pytorch.org/libtorch/cu100/libtorch-win-shared-with-deps-1.1.0.zip
        https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.1.0.zip

2. Install cuDNN
3. Unpack libtorch (e.g. to `supra/external` on windows or `/opt/` on linux)
4. Activate `SUPRA_TORCH` in cmake (e.g. in the GUI, or via `-DSUPRA_TORCH=ON`)
5. Point cmake to the libtorch you just extracted (e.g. `supra/external/libtorch/share/cmake/Torch` or `/opt/libtorch/share/cmake/Torch`)
6. Configure and build
=======
	
Demo (No US-system required!)
----------------

Change to your build directory. If you used the commands above, you can execute

	cd $HOME/git/supra/build

Start the SUPRA GUI with a demo config file

	src/GraphicInterface/SUPRA_GUI -c data/configDemo.xml -a
	
Where `-c` defines the config file to load and `-a` is autostart.

This shows a complete ultrasound pipeline running on your computer from raw channel data recorded with
a Cephasonics system and a 7MHz linear probe.
With the dropdown menu "Preview Node", you can select which stage of the pipeline to inspect.
For the final state of the image, select "SCAN", which shows the output of the scan-converter - the B-mode.
	
Used libraries
----------------

**SUPRA** uses tinyxml2 which is awesome and distributed under the zlib-license. For more details see the [tinyxml2 README](src/SupraLib/utilities/tinyxml2/readme.md) and (http://grinninglizard.com/tinyxml2/index.html and https://github.com/leethomason/tinyxml2)

**SUPRA** also uses jsoncpp for more structured data handling which is distributed under the MIT license. For more details see the [jsoncpp README](src/SupraLib/utilities/jsoncpp/README.md)

On windows, ROS-message headers generated with [rosserial](http://wiki.ros.org/rosserial) are used and are included in the source.
On Linux, the usual ROS-libraries are used during build. (roscpp, geometry_msgs)

**SUPRA** additionally uses the Intel Thread Building Blocks (but does not provide them) in their Apache 2.0 licensed form. https://www.threadingbuildingblocks.org/

Finally, it can be built against
	
* QT (LGPLv3)
* IGTL (BSD 3clause)
* CAMPVis (Apache 2.0) (unfortunately, the respective QT5 version is not yet public)

### Alternate Builds

REST interface instead of graphical interface
----------------

Build requirements

	apt-get install cmake cmake-gui libtbb-dev libopenigtlink-dev libcpprest-dev libboost-all-dev git
	
SUPRA

	mkdir -p $HOME/git && cd $HOME/git #(or your favorite directory for repositories)
	git clone https://github.com/IFL-CAMP/supra.git
	cd supra
	mkdir -p build && cd build
	cmake-gui .. -DSUPRA_INTERFACE_REST=ON -DSUPRA_INTERFACE_GRAPHIC=OFF

1. Configure
2. For systems with multiple gcc versions, make sure to select one supported by the installed CUDA version
3. You might need to specify the CUDA toolkit directory (usually "`/usr/local/cuda`")
4. Configure & Generate, then close cmake and build
5. Build SUPRA
	
	make -j5
	
6. Start SUPRA: See below

#### Rest Interface Queries

SUPRA accepts GET and POST requests.

The IP address / hostname SUPRA can be reached with is referred as `SUPRA_URL` below.

##### GET REQUESTs

`SUPRA_URL/nodes/[var]` where var can be `input` to return all input nodes, `output` to get only the output nodes and empty or `all` to return all nodes regardless of their types.
The shape of the object in response's body will be 
`{"nodeIDs":[String]}`.

`SUPRA_URL/parameters` returns all parameters for one node.
The shape of the object to send with the body is `{"nodeID":"ID"}`.


##### POST REQUESTs

`SUPRA_URL/parameters` sets the value of a parameter of a node. The request has to be shaped like below.

	{
		"nodeID":"id",
		"parameterID":"id",
		"value":"value"
	}


Demo (No US-system required!)
----------------

Change to your build directory. If you used the commands above, you can execute

	cd $HOME/git/supra/build

Start the SUPRA GUI with a demo config file

	src/RestInterface/SUPRA_REST data/configDemo.xml
	
Additionaly used libraries
----------------
See above for most used libraries. This build uses additionally:
* Microsoft C++ Rest SDK >=2.8 - (BSD 3clause)
* Boost (MIT)

Generate a self-building deb source file
----------------

Build Requirements:

	apt-get install debmake

	cd supra
	debmake -cc >> copyright
	mkdir -p build && cd build
	cmake ..
	make package_source

The deb file can be found in the 'binpackages' folder.

When installing the deb file in a system the package will try to build with the standard cmake configuration on that system.

Acknowledgement
----------------

SUPRA logo by Raphael Kretz.

![EU flag](http://campar.in.tum.de/files/goeblr/EUflag.png "EU flag")

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No. 688279.
