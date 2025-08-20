# Robust Dexterous Grasping of General Objects


## [Paper](https://arxiv.org/abs/2504.05287) | [Project Page](https://zdchan.github.io/Robust_DexGrasp/) 

<img src="/RobustDexGrasp_application.gif" autoplay loop /> 

### Contents

1. [Info](#Info)
2. [Installation](#installation)
3. [Demo](#Demo)
4. [Citation](#citation)
5. [License](#license)



## Info

The repository comes with all the features of the [RaiSim](https://raisim.com/) physics simulation. The RobustDexGrasp related code can be found in the [raisimGymTorch](./raisimGymTorch) subfolder. The policies are trained and evaluated on the Allegro+UR5 platform.

There are 2 environments (see [envs](./raisimGymTorch/raisimGymTorch/env/envs/)) for training and evaluation in simulation, one for teacher policy and one for student policy. Each environment contains a configuration file, a C++ based simulation environment, a training script, a visualization script, and a quantitative evaluation script. We also provide the environment for hardware deployment. The pre-trained policies are saved in [data_all](./raisimGymTorch/data_all).



## Installation


For good practice for Python package management, it is recommended to use virtual environments (e.g., `virtualenv` or `conda`) to ensure packages from different projects do not interfere with each other. The code is tested under Python 3.8.10.

### RaiSim setup

RobustDexGrasp is based on RaiSim simulation. For the installation of RaiSim, see and follow our documentation under [docs/INSTALLATION.md](./docs/INSTALLATION.md). Note that you need to get a valid, free license for the RaiSim physics simulation and an activation key (run any script and follow the instruction).

### RobustDexGrasp setup

After setting up RaiSim, the last part is to set up the RobustDexGrasp environments.

```
$ cd raisimGymTorch 
$ python setup.py develop
```

All the environments are run from this raisimGymTorch folder. 

Note that every time you change the environment.hpp, you need to run `python setup.py develop` again to build the environments.

Then install pytorch with (Check your CUDA version and make sure they match)

```
$ pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

Install required packages

```
$ pip install scipy
$ pip install scikit-learn scipy matplotlib
```

You should be all set now. Try to run the demo!

### Other alternative requirements

1. (Only for hardware deployment) You should follow the doc in [real_setup.md](./real_setup.md).
2. If you want to test/grasp your own objects in simulation, we released the script in [GraspXL](https://github.com/zdchan/GraspXL) to pre-process new objects. Clone the folder [URDF_gen_from_obj](https://github.com/zdchan/GraspXL/tree/main/raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj) into [helper](./raisimGymTorch/raisimGymTorch/helper/). Put the .obj files you want to grasp (make sure they have meaningful sizes for grasping) under [URDF_gen_from_obj/temp](.https://github.com/zdchan/GraspXL/tree/main/raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj/) and run [urdf_gen.py](https://github.com/zdchan/GraspXL/tree/main/raisimGymTorch/raisimGymTorch/helper/URDF_gen_from_obj/urdf_gen.py), then it will generate a folder with the processed objects and the urdf files in [rsc](./rsc), which you can further utilize with any environment scripts.


## Demo

We provide some pre-trained models to view the output of our method. They are stored in [this folder](./raisimGymTorch/data_all/). 

+ For interactive visualizations, you need to run

  ```Shell
  ./../raisimUnity/linux/raisimUnity.x86_64
  ```

  and check the Auto-connect option.

+ To randomly choose an object with random poses and visualize the grasping in simulation (use student policy as an example), run

  ```Shell
  python raisimGymTorch/env/envs/allegro_student/visual_eval.py
  ```

You can indicate the objects or the objectives of the generated motions in the visualization environments

+ The object is by default a random object from the training set, which you can change to a specified object by the variable obj_item in visualization scripts. You can also specify the object set by the variable cat_name for both visualization and evaluation scripts. 



## BibTeX Citation

To cite us, please use the following:

```bibtex
@inproceedings{zhang2025RobustDexGrasp,
    title={{RobustDexGrasp}: Robust Dexterous Grasping of General Objects},
    author={Zhang, Hui and Wu, Zijian and Huang, Linyi and Christen, Sammy and Song, Jie},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2025}
  }
```

If you utilize the object pre-process script provided by GraspXL for your own objects, please consider cite:

```bibtex
@inProceedings{zhang2024graspxl,
  title={{GraspXL}: Generating Grasping Motions for Diverse Objects at Scale},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Hilliges, Otmar and Song, Jie},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```



## License

This work and the dataset are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
