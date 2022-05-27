# Histogram-based Video Shot Transition Detection
This code detects abrupt (cut) shot transitions in a given input video by, hierarchical temporal partitioning of video frames, using block-color histogram of the consequitive frames, and thresholding. This code is written based on the cut transition detection method that is introduced in these papers: [paper1](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7585511), [paper2](https://arxiv.org/pdf/2104.10847.pdf).
<p align="center">
  <img width="400" src="https://github.com/MehrnazFani/Shot-Transition-Detection/blob/e4799ace3d68f7f4e7cc886e8d0e509ee83213f8/img-for-readme/CT-soccer.png" alt="CT detection">
</p>

<p align="center">
   <img width="600" src="https://github.com/MehrnazFani/Shot-Transition-Detection/blob/61a696880f5e69bca3e86c296006f88153c3a45e/img-for-readme/TemporalPartitioning.png" alt=" Hierarchical partitioning of video frames">
</p>

## Environment
+ Ubuntu 18.04
+ CUDA 10.0
+ CuDNN 10.0
+ Python 3.8
+ Anaconda 3: Use "requirement.txt" to create a conda environment with all required packages
```
conda create --name <env> --file requirements.txt
```
## Using the code

+ **Step 1:** Put the input video files (<video_name>.mp4) in "videos" folder.
+ **Step 2:** Run [shot_transition_detection.py](https://github.com/MehrnazFani/Shot-Transition-Detection/blob/7687a09197d2b0a51074024fe5e9d540273d93b0/shot_transition_detection.py)

+ **Outputs:** A folder will be created, i.e. "./videos/<video_name>" , that will include:
>  - All video shots, ".mp4".
> -  A “.csv” file with information about all of the generated video shots.
>
# Cite us please
Please cite the following papers if you are using this code
```
@inproceedings{yazdi2016shot,
  title={Shot boundary detection with effective prediction of transitions' positions and spans by use of classifiers and adaptive thresholds},
  author={Yazdi, Mehran and Fani, Mehrnaz},
  booktitle={2016 24th Iranian Conference on Electrical Engineering (ICEE)},
  pages={167--172},
  year={2016},
  organization={IEEE}
}


@article{fani2021localization,
  title={Localization of Ice-Rink for Broadcast Hockey Videos},
  author={Fani, Mehrnaz and Walters, Pascale Berunelle and Clausi, David A and Zelek, John and Wong, Alexander},
  journal={arXiv preprint arXiv:2104.10847},
  year={2021}
}
``` 
