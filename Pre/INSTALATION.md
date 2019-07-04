# INSTALATION

# HOW TO INSTALL BLENDER (LINUX) 
(full version https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu)

## 1 Get the source
    
    mkdir ~/blender-git
    cd ~/blender-git
    git clone https://git.blender.org/blender.git
    cd blender
    git submodule update --init --recursive
    git submodule foreach git checkout master
    git submodule foreach git pull --rebase origin master

- ### 1.1 If you want update Blender's source code to the latest development version. So in ~/blender-git/blender/
    
    make update
## 2. Install/Update the dependencies
  - ### 2.1 Recommended to use install_deps.sh script. To use it you are only required to install the following dependencies:

        git, build-essential 
        for git: sudo apt-get install git
        for build-essential: sudo apt-get install build-essential

  - ### 2.2 Then, get the sources and run install_deps.sh

        cd ~/blender-git
        ./blender/build_files/build_environment/install_deps.sh

## 3. Compile Blender with CMake
  - ### 3.1 Installing CMake
  
         sudo apt-get -y install cmake
    
  - ### 3.2 Automatic CMake Setup
  
        cd ~/blender-git/blender
        make
    
    - #### Once the build finishes you'll get a message like..

          Blender successfully built, run from: /home/me/blender-git/build_linux/bin/blender
    
  - ### 3.3 Updating your local checkout and rebuilding is as simple as:
   
         cd ~/blender-git/blender
         make update
         make    
    
 # FINISH
 
 ### BUT if you want Edit CMake Parameters go to https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu
    

# To run the script from terminal
(more info: https://learnsharewithdp.wordpress.com/2018/08/27/how-to-run-a-python-script-in-blender/)

    blender filename.blend --python script.py in our case
    blender ocean_render_2.blend --python 4macro.py
  
