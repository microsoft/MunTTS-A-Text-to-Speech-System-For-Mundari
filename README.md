# MunTTS: A Text-to-Speech System for Mundari

**Official Repository for ["MunTTS: A Text-to-Speech System for Mundari"](https://aclanthology.org/2024.computel-1.11/). This work has been done in collaboration with [Karya](https://karya.in/).**

## Abstract
We present **MunTTS**, an end-to-end text-to-speech (TTS) system specifically for Mundari, a low-resource Indian language of the Austo-Asiatic family. Our work addresses the gap in linguistic technology for underrepresented languages by collecting and processing data to build a speech synthesis system. We begin our study by gathering a substantial dataset of Mundari text and speech and train end-to-end speech models. We also delve into the methods used for training our models, ensuring they are efficient and effective despite the data constraints. We evaluate our system with native speakers and objective metrics, demonstrating its potential as a tool for preserving and promoting the Mundari language in the digital age.

## Setup
Follow the given commands to create a virtual environment for this project and install necessary packages. 

- Create an experiment directory 
- Clone this repository in it and run the following commands (preferable to use a Linux-based system with a GPU)

```bash 
# create a virtual environment (preferably, use Python 3.10+)
conda create -n coqui-tts python=3.10
conda activate coqui-tts

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pymcd scikit-learn ffmpeg

# install Coqui-TTS
git clone https://github.com/coqui-ai/TTS
pip install --editable ./
cd ../

# install indic-nlp-library
git clone https://github.com/VarunGumma/indic_nlp_library
cd indic_nlp_library
pip install --editable ./
cd ../
```

## Directory Structure
```bash
.
├── README.md
└── src                                   # main directory with all the scripts and files
    ├── compute_mcd.py                    # Python script to compute MCD scores
    ├── format_dataset.py                 # Python script to format the dataset into the required structure
    ├── scripts                           # directory for helper shell scripts
    │   ├── audio_enhance.sh              # helper script to clean and enhance audio using ffmpeg
    │   ├── compute_mcd.sh                # helper script to compute MCD scores
    │   ├── finetune_xtts.sh              # helper script to finetune XTTS
    │   ├── format_dataset.sh             # helper script to format the dataset
    │   ├── test_mms.py                   # Python script to generate speech from MMS
    │   ├── test_mms.sh                   # helper script to generate speech from MMS
    │   ├── test_vits.sh                  # helper script to generate speech from VITS                    
    │   ├── test_xtts.sh                  # helper script to finetune XTTS
    │   └── train_vits.sh                 # helper script to train VITS
    ├── utils.py                          # utilities file to store common functions
    ├── vits.py                           # Python script to train VITS 
    └── xtts.py                           # Python script to finetune XTTS
```

## Dataset (Released under the Karya License (BY-NC-SA-FS 1.0))
_Please contact Karya data resources (data@karya.in) for the full dataset, usage and distribution._

## Model Checkpoints (Released under the Karya License (BY-NC-SA-FS 1.0))
_Please contact Karya data resources (data@karya.in) for the VITS model checkpoints, usage and distribution._


## MOS Values
| Model               | Full ($100$)    | Male ($26$)    | Female ($74$)    |
|---------------------|-----------------|----------------|------------------|
| *gt-22k*            | 4.62±0.68       | 4.59±0.65      | 4.63±0.69        |
| *gt-44k*            | 4.58±0.70       | 4.47±0.79      | 4.62±0.66        |
| *mms*               | 0.79±1.02       | 0.79±1.02      | $-$              |
| *vits-22k*          | 3.04±1.29       | 2.65±1.34      | 3.18±1.25        |
| *vits-44k*          | **3.69±1.18**   | **3.39±1.25**  | **3.79±1.13**    |
| *xtts-finetuned*    | 0.05±0.30       | 0.13±0.52      | 0.02±0.16        |
| *xtts-pretrained*   | 2.20±1.32       | 2.10±1.36      | 2.23±1.31        |

## Training a new VITS model
- Run `src/format_dataset.sh` with necessary cmdline arguments to format your dataset in the required fashion. 
- Run `src/train_vits.sh` with necessary cmdline arguments to train a VITS model on your data. It is recommended you adjust the hyperparameters as per your requirement. 
- Run `src/test_vits.sh` with necessary cmdline arguments to generate speech samples for the test set using the best checkpoint of the VITS model just trained. This is also apply audio enhancement using `ffmpeg` to generated and the enhanced audios are saved in a directory with suffix `_cleaned`.

## Citation
In you use this dataset, code-base or models, please cite our work,
```bibtex
@inproceedings{gumma-etal-2024-muntts,
    title = "{M}un{TTS}: A Text-to-Speech System for {M}undari",
    author = "Gumma, Varun  and
      Hada, Rishav  and
      Yadavalli, Aditya  and
      Gogoi, Pamir  and
      Mondal, Ishani  and
      Seshadri, Vivek  and
      Bali, Kalika",
    editor = "Moeller, Sarah  and
      Agyapong, Godfred  and
      Arppe, Antti  and
      Chaudhary, Aditi  and
      Rijhwani, Shruti  and
      Cox, Christopher  and
      Henke, Ryan  and
      Palmer, Alexis  and
      Rosenblum, Daisy  and
      Schwartz, Lane",
    booktitle = "Proceedings of the Seventh Workshop on the Use of Computational Methods in the Study of Endangered Languages",
    month = mar,
    year = "2024",
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.computel-1.11",
    pages = "76--82",
    abstract = "We present MunTTS, an end-to-end text-to-speech (TTS) system specifically for Mundari, a low-resource Indian language of the Austo-Asiatic family. Our work addresses the gap in linguistic technology for underrepresented languages by collecting and processing data to build a speech synthesis system. We begin our study by gathering a substantial dataset of Mundari text and speech and train end-to-end speech models. We also delve into the methods used for training our models, ensuring they are efficient and effective despite the data constraints. We evaluate our system with native speakers and objective metrics, demonstrating its potential as a tool for preserving and promoting the Mundari language in the digital age.",
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
