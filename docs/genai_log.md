## AI Use

### Brant
- 11/13 at 10:47AM: Brant had Claude (Sonnet 5.5 medium) make a copy-pasteable tree that reflects our outlined folder structure for this project to be added to the README. We did this because I didn't want to type out all the dashes and branches ourselves
- 11/13 at 11:30AM: Brant had Claude (Sonnet 5.5 medium) draft the recommended function structure for the data.py processing pipeline. This allows us to make sure the pipeline  isn't missing any steps. This pipeline is tested by checking output for overlap in train, val, and test splits and checking the proportion of toxic labels
- 11/13 at 12:17AM: Brant had Claude (Sonnet 5.5 medium) draft a script for testing config.py and data.py. The script is test/check_splits.py. Did this to assist in verifying and debugging code before moving to next steps
- 11/13 at 1:10PM: Brant had ChatGPT draft the function structure for baseline.py to create an efficient structure for the code. Also had it draft the app.py framework because this is my first time working with gradio. Tested by changing some of the fields and relaunching the app to see what changes as well as confirming app functionality with the model from baseline.py 
### Garrett
- 11/13, morning: Garrett had Claude Code (Claude Opus 5.5) draft the identity term list and the 120 template sentences used to build our 2,400-sentence audit set. Garrett then read through and checked the templates and terms before using them.
- 11/13, morning: Garrett had Claude Code draft the counterfactual augmentation script and the DistilBERT training and scoring code, based on the settings in our proposal. Garrett reviewed this code and tested it on sample inputs.
- 11/13, morning: Garrett had Claude Code draft the bias audit script and the demo case script, which compute our fairness metrics and score the demo examples. Garrett went through the scripts and tested their outputs to make sure the metrics were calculated correctly.
- 11/13 at 11:40AM: Garrett ran the full pipeline on a small fake dataset with a tiny model and used Claude Code to help debug the errors that came up, including a training setting newer library versions no longer support. Garrett reran the pipeline afterward to confirm the fixes worked.
- 11/13 at 11:47AM: Garrett had Claude Code write the commit message and open a pull request for his code, and help debug a GitHub login error on his laptop. Garrett checked the pull request before it was merged.
- 11/15: Garrett used Claude Code to help debug the Colab notebook so it could access our private repo and run the full pipeline. Garrett tested the notebook in Colab before using it for training.
