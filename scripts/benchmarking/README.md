**This folder contains the code relating to the evaluation section of this project**

The files have the following content:
- ```qqp_corrs.py```: Compute the performance by metric on the Quora Question Pairs task
- ```get_corrs.py```: Compute the correlation between the metrics and human judgement scores on the annotated CNN/DailyMail dataset
- ```adversarial.py```: Corrupt a set of summaries and assess the performance of the metrics at discriminating between corrupted and uncorrupted summaries
- ```benchmark.py```: File for performing evaluation of summaries. Check main README for full usage. Example shell script is in ```sh_scripts```
- ```resource.py```: utils
- The remaining folders contain auxilliary code for running the above files
