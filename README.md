# tattooed
Give your AI model a nice tattoo! 🦾  Or, a new *permanent* watermarking technique.

This repository contains the source code related to the paper **TATTOOED: A Robust Deep Neural Network Watermarking Scheme based on Spread-Spectrum Channel Coding**.
In this paper, we build upon extensive prior work on covert (military) communication and propose TATTOOED, a novel DNN watermarking technique that is robust to existing threats. 
The code is intended to be as modular as possible to allow for extension of this work to other datasets, model architectures and more.
The code will be rendered public permanently through a Github repository for future researchers and enthusiasts of this area of work. 
As we quote even in the end of this document, TATTOOED was made with love, and we are more than happy to share it with the world. 
In what follows a guideline on how to setup and run different experiments included in the paper is provided.

## Requirements

### System Information

The experiments are run on a device running Ubuntu 20.04.6 LTS operating system. This device was equipped with 64GB of RAM, a Ryzen 7 2700x 8 core processor and an Nvidia RTX 2080 Ti GPU with 11GB of memory.

**Note:** The amount of computational power to run TATTOOED is well below the computational power of the abovementioned setup. For example, running experiment using one of the largest DNN models namely VGG requires only 4.2GB of GPU memory. This value is for the training of the model. For watermarking the model can be loaded on RAM and the watermarking can be done on CPU also.
For a smooth running of TATTOOED we recommend a system with at least 16GB of RAM memory, 4-8 core CPU and at least a terabyte of storage (this is just because datasets such as Imagenet are circa 180 GB)

### Dependencies

Once you are in the `tattooed` folder, we recommend creating a virtual environment, ensuring that the code runs smoothly without interfering with your main Python installation. 

**Virtual environment:** Create a virtual environment using `venv`

```bash
# if virtualenv package is not already installed
pip install virtualenv

# create a virtual environment under .venv folder
virtualenv -p python3.8 tattooedvenv

# activate your virtual environment
source tattooedvenv/bin/activate

# install project requirements
pip install -r requirements.txt

# NOTE: For some unknown reason in some cases pip throws an error while running the abovementioned command. It was brought to our attention that by first installing the numpy package before by running "pip install numpy" and then following up with the abovementioned command solves this issue.

```


### Project structure

The project structure is like this:

- There is only one python file in the main folder - tattooed.py - which will be used to run the experiments.
---
The folders:
- checkpoints
  - In this directory there will be stored the trained model files
- config
  1. dataset - contains the dataset information and some necessary infor for training such as batch size , number of classes etc. One .yaml file per dataset.
     - cifar10.yaml
     - mnist.yaml
     - ...
  2. model - contains the model architecture information such as name, optimizer to use etc. One .yaml file per architecture.
      - mlp.yaml
      - vgg.yaml
      - ...
- data - the folder where the training/ testing data for each task is stored.  Most downloads are handled by torchvision itself.
- dataset - this folder contains the definition of the torchvision dataloader for each considered dataset. One file per dataset:
  - mnist.py 
  - cifar10.py 
  - ...
- logger - contains the configuration for the information logging such as the model performance each epoch etc.
  - csv_logger.py 
  - ...
- marks - the folder where we store the watermarks to choose from.
   - extract - in this folder it will be stored the extracted watermark content after each run so one can visually verify
   - mark.txt - textfile containing the TATTOOED watermark! text watermark as shown in the paper.
   - ...
- models - model architecture definitions. One .py file per model architecture.
   - mlp.py
   - vgg.py
   - resnet.py
   - ...
- utils - some utility files containing different methods used in the project
   - utils_bit.py - utility methods to read write bit sequences to files 
   - utils_nn.py - utility methods for a neural network such as pruning etc.
   - ...
- watermark - contains the callback definition that we use to schedule the embedding and extraction of TATTOOED watermark and the actual definition of **mark** and **verify** procedures.
   - callback.py
   - watermark.py
   - ...
- outputs - stores the details about the run performance and watermarking procedures during each experiment. For each dataset there will be created a folder which will contain various folders with the model architecture name. Inside there will be timestamped folders which will contain logged run information as follows:
   - cifar10
     - vgg
       - 2024-08-26_11-51-30
         - tattooed.log
         - train.csv
         - val.csv
   - ...
   - plot_accuracy.py - python file to use for plotting the accuracy values such as plots of Figure 2 and 3 of the paper.
### Data
* MNIST, CIFAR10, CIFAR100 datasets are already available via Torchvision.
* GTSRB, PETS and Imagenet datasets should be downloaded and placed under `/data` folder. Links to the datasets are found below:
  - GTSRB: [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
  - Imagenet: [https://image-net.org/download-images](https://image-net.org/download-images)
  - PETS: [https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip)

## Running TATTOOED
You can modify the `config.yaml` file located in the `config` folder to change the default settings, and then run:

    python tattooed.py 

In what follows we list the different configurations to run different types of experiments.

### Example experiments config files

1. Regular model training and watermarking.
   - This corresponds to the case when one wants to watermark their model prior to release. Here you can observe the effect of the watermark on the TER (test error rate), i.e., prior and post watermarking performance of a model in the intended task.
   - Example case: MNIST dataset, MLP architecture, *text* watermark. 
   - Precondition: None, this is the base case.
   - One epoch of mnist on mlp is around 8-10 seconds on our setup, one epoch on cifar10 on vgg requires circa 35 seconds on our setup.
   ```yaml
        trainer:
          train_epochs: 60
          progress_bar_refresh_rate: 1
          gpus: 1
          fine_tuning: false
          model_name: ""
        
        watermark:
          seed: 42
          ldpc_seed: 8
          parameter_seed: 43
          ratio: 0.75
          error_correction: true
          embed_path: "marks/mark.txt"
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
                   
        removal:
          prune: false
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: mnist
          - model: mlp
   ```
   - Note: modify the dataset name and model for other examples such as VGG on Cifar10 dataset. On such large models the **ratio** parameters should be set to a very low value such as 0.0125. As we state in the paper, one does not necessarily need to use all the weight parameters of the model for watermarking, a random subsample would suffice.  
2. Finetuning an already watermarked model RTAL or FTAL
   - This corresponds to the case when a model is watermark but is later fine-tuned on the same task either using same data or new data. Experimentally this consists of either splitting the main dataset in two equal parts for (RTAL) or following a specific ratio like 80/20 for FTAL. 
   - Example case: MNIST dataset, MLP architecture, *text* watermark. This will load the already trained model and finetune it on the MNIST dataset for the specified number of epochs and then check whether the watermark is there.
   - Precondition: An already watermarked model.
   - One epoch of mnist on mlp is around 8-10 seconds on our setup, one epoch on cifar10 on vgg requires circa 35 seconds on our setup. So the RTAL experiment shown on Figure 2a totaling 1100 (100 training + 1000 fine-tuning) epochs would require circa 3 hours. The 2b circa 10.5 hours.
   ```yaml
        trainer:
            train_epochs: 60
            progress_bar_refresh_rate: 1
            gpus: 1
            fine_tuning: true
            model_name: "model_mlp_dataset_mnist.pt"
        
        watermark:
          seed: 42
          ldpc_seed: 8
          parameter_seed: 43
          ratio: 0.75
          error_correction: true
          embed_path: "marks/mark.txt"
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
                   
        removal:
          prune: false
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: mnist
          - model: mlp
   ```
3. Finetuning using a different dataset.
   - This considers a situation when you download an already trained model (and watermarked in this case), replace the last (output) layer to solve a new but similar task.
   - Example scenario: VGG architecture trained on Cifar10 dataset and watermarked using the *text* watermark is loaded and fine-tuned on GTSRB dataset. This will load the already trained model and finetune it on the GTSRB dataset for the specified number of epochs and then check whether the watermark is there.
   - Precondition: An already pretrained and watermarked model.
   -    For the FTAL with different dataset shown on figure 3a and 3b: ResNet-18 is loaded from torch already trained on Imagenet, watermarked and then fine-tuned on the Cats vs. Dogs dataset for 60 epochs each requiring circa 55 seconds on our setup (circa one hour on our setup). Regarding the figure 3b you need to train a vgg model on cifar10 for 60 epochs and watermark it (this entire procedure requires around 40-45 minutes on our setup), then finetune it on the GTSRB task for another 60 epochs requiring circa 40-45 minutes on our setup too.

   ```yaml
        trainer:
            train_epochs: 60
            progress_bar_refresh_rate: 1
            gpus: 1
            fine_tuning: true
            model_name: "model_vgg_dataset_cifar10.pt"
        
        watermark:
          seed: 42
          ldpc_seed: 8
          parameter_seed: 43
          ratio: 0.0125
          error_correction: true
          embed_path: "marks/mark.txt"
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
           
        removal:
          prune: false
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: gtsrb
          - model: vgg
    ```
4. Model weight parameter pruning.
   - This considers a situation when some weight parameters are pruned (i.e., zeored) attempting to remove the watermark.
   - Example scenario: MLP architecture trained on MNIST task for 60 epochs, watermarked using the *text* watermark and one adversary might try to prune the weights in an attempt to disrupt the watermark.
   - Precondition: A trained and watermarked model, or you can train (and watermark) from scratch and the pruning will take place after.
   ```yaml
        trainer:
            train_epochs: 60
            progress_bar_refresh_rate: 1
            gpus: 1
            fine_tuning: false
            model_name: ""
        
        watermark:
          seed: 42
          ldpc_seed: 8
          parameter_seed: 43
          ratio: 0.75
          error_correction: true
          embed_path: "marks/mark.txt"
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
           
        removal:
          prune: true
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: mnist
          - model: mlp
    ```
Table 4 in the paper corresponds to pruning experiments on different pruning ratios from 25% up to 99.99%. So the data in Table 4 come from running this multiple times each times using a different pruning ratio as shown in the first column of Table 4 [25%, 50%,  75%, 90%, 95%,  99%, 99.75%, 99.99%] and report the corresponding TER and BER that you can see as output on the terminal. Each pruning run takes around 1-2 minutes on our setup including both the evaluation on the test set and the verification of the watermark presence on the network.


5. REFIT - the watermark removal attack by Chen et al.
   - This part explains about the usage of REFIT attack. The paper is found at [link](https://arxiv.org/abs/1911.07205) and the code (public) by the authors can be found at this [Github](https://github.com/sunblaze-ucb/REFIT).
   - REFIT is not part of this repository as it is not our work but below we list how we run it:
     1. Clone the public repository provided by the authors of REFIT and setup using their requirements.
     2. Train and watermark a model as shown in point 1 of this list.
     3. Take the saved checkpoint and run the REFIT attack as shown [here](https://github.com/sunblaze-ucb/REFIT)
     4. Evaluate the presence of the watermark in the model using the source code we have provided.
     5. **Note:** Table 3 (in the paper) presents the results of this experiment in binary format. Checkmark for **watermark survived the attack** and X for not survived. 


6. Plotting the accuracy data such as Figure 2 and 3 in the paper:
   - To plot the model performance such as the ones shown on figures 2 and 3 both for RTAL, FTAL or any kind of training use the **plot_accuracy.py** file located in the **/outputs** directory
   - You will need to pass as command line argument the path to the tran.csv or val.csv file and the value of the epoch you injected the watermark for the yellow vertical line. You can find this either at the config.yaml you used for that run or at the tattooed.log file that will be under the same folder as the other .csv files of that run.
```bash
python plot_accuracy.py dataset_name/model_name/timestamp/train.csv 60
```


7. Parameter shuffling and watermark overwriting:
   - **Parameter shuffling** is a theoretical result on how you can unshuffle a mixed set of data of a matrix following mathematical theory thus there is no code for that.
   - **Watermark Overwriting** this experiment regards the case when an adversary takes an already watermarked model and tries to add one or more other watermarks in an attempt to overwrite the legitimate owners watermark. In practice this is achieved by running the same configuration twice where in the second time you change the **seed**, **ldpc_seed** **parameter_seed**. Basically you change the secret key (all those three parameters simulate the real life secret key) that is known solely from the legitimate owner and the adversary can not find out. As we have highlighted in the paper, the secret key can not be guessed. After the second run you revert back to the initial seed values and run the code by just commenting out the trainer.fit(model, data) on line 146 and leaving only the success = watermarker.extract(model) on line 157 of tattooed.py.
   - **Watermark Overwriting** example:
   1. Step 1:
   ```yaml
        trainer:
            train_epochs: 60
            progress_bar_refresh_rate: 1
            gpus: 1
            fine_tuning: false
            model_name: ""
        
        watermark:
          seed: 42
          ldpc_seed: 8
          parameter_seed: 43
          ratio: 0.75
          error_correction: true
          embed_path: "marks/mark.txt"
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
      
        removal:
          prune: false
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: mnist
          - model: mlp
    ```
   
   2. Step 2: Change the watermark seeds to anything besides the initial seeds (the adversary can not know or neither guess them). As we state in the paper, the secret key is 512 bits and as such we wish the adversary good luck into trying to guess it in their lifetime. After this step the model will contain another watermark besides the one from the legitimate owner.
   ```yaml
        trainer:
            train_epochs: 60
            progress_bar_refresh_rate: 1
            gpus: 1
            fine_tuning: true
            model_name: "model_mlp_dataset_mnist.pt"
        
        watermark:
          seed: 879
          ldpc_seed: 98
          parameter_seed: 6543
          ratio: 0.75
          error_correction: true
          embed_path: "marks/some_other_mark.txt" 
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
           
        removal:
          prune: false
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: mnist
          - model: mlp
    ```
   3. Step 3: Change the watermark seeds back to the original (step 1). Leave the fine-tuning and model name in order to load the same model. In **tattooed.py** comment out the line 146 (model.fit(model, data)) such that no further training is carried out but just a model evaluation and then watermark extraction. You will still be able to see the initial watermark extracted with success after this step. We bring to the attention of the reader that robustness to overwriting comes by default upon the reliance on CDMA and it is proven theoretically since the 1950 in the digital communication setup. This experiment here shows that in practice. Basically you can assume two entities communicating using the same channel using different spreading codes. CDMA guarantees that their corresponding receiver will get the correct message without distorting the others.
   ```yaml
        trainer:
            train_epochs: 60
            progress_bar_refresh_rate: 1
            gpus: 1
            fine_tuning: true
            model_name: "model_mlp_dataset_mnist.pt"
        
        watermark:
          seed: 42
          ldpc_seed: 8
          parameter_seed: 43
          ratio: 0.75
          error_correction: true
          embed_path: "marks/mark.txt" 
          extract_path: "marks/extract/"
          start: 59
          end: 60
          gamma: 0.0009
        
        hydra:
          run:
            dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
          job:
            chdir: True    
                   
        removal:
          prune: false
          prune_ratio: 0.1
        
        defaults:
          - _self_
          - dataset: mnist
          - model: mlp
    ```

   8. Extending TATTOOED to new model architectures:
      - TATTOOED relies on code-division multiple-access to robustly watermark a DNN model without imparing the performance on its legitimate task. Practically, this reliance on CDMA, coupled with the code organization we provide renders it easy to extend TATTOOED to new architectures. In what follows we explain how this can be done. There are 4 steps to be followed to include a new task and new model architecture completely following the structural organization we present in this repository:
          1. **Step 1** Create the dataset and model config files (to be put under config/dataset and config/model directory):
             - The dataset config is a short .yaml file containing information about the dataset such as name, number of output classes, preferred batch size etc. - The model config is another short .yaml file containing information about the model such as name, preferred optimizer, default learning rate etc as follows:

          ```yaml
                name: cifar100
                dim: 32
                n_classes: 100
                batch_size: 64
                num_workers: 10
         ```
         ```yaml
             name: mlp
             optimizer: "adam"
             lr: 3e-4
         ```
        2. **Step 2** Create the dataloader for teh dataset. This comprises writing few lines of python that essentially load and organize the data to be used for training and testing your model. Most of the code is standard as per Torch guidelines. We present the example of MNIST dataset dataloader here. For each other dataset (the raw data need to be put under data/ directory for better organization) you would need to edit this code accordingly.
          ```python
           from typing import Optional
           from torch.utils.data import DataLoader, random_split
           from torchvision import datasets, transforms
           import pytorch_lightning as pl
          
           class MNIST(pl.LightningDataModule):
                def __init__(self, base_path, batch_size=64, num_workers=1):
                    super().__init__()
                    self.base_path = base_path
                    self.batch_size = batch_size
                    self.num_workers = num_workers
            
                def prepare_data(self):
                    # download only
                    datasets.MNIST(self.base_path / 'data', train=True, download=True, transform=transforms.ToTensor())
                    datasets.MNIST(self.base_path / 'data', train=False, download=True, transform=transforms.ToTensor())
            
                def setup(self, stage: Optional[str] = None):
                    # transform
                    transform = transforms.Compose([transforms.ToTensor()])
                    mnist_train = datasets.MNIST(self.base_path / 'data', train=True, download=False, transform=transform)
                    mnist_test = datasets.MNIST(self.base_path / 'data', train=False, download=False, transform=transform)
            
                    # train/val split
                    mnist_train, mnist_val = random_split(mnist_train, [50000, 10000])
            
                    # assign to use in dataloaders
                    self.train_dataset = mnist_train
                    self.val_dataset = mnist_val
                    self.test_dataset = mnist_test
            
                def train_dataloader(self):
                    return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            
                def val_dataloader(self):
                    return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            
                def test_dataloader(self):
                    return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        ```
   
      3.  **Step 3** After creating the appropriate dataloader, you need to obviously define your new model architecture. The model definition file needs to be put under models/ directory. In the model file you need to define the train, validation adn test methods, plus the base definition of the DNN architecture. You can take as a base example any of the architecture definitions found under the models/ directory. The general skeleton looks as follows.
        ```python
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            import pytorch_lightning as pl
            from torchmetrics.functional.classification import accuracy
            
            
            class MLP(pl.LightningModule):
                def __init__(self, input_size, num_classes, optimizer='adam', learning_rate=2e-4):
                    super().__init__()
            
                    self.layer_1 = nn.Linear(input_size * input_size, 128)
                    self.layer_2 = nn.Linear(128, 256)
                    self.layer_3 = nn.Linear(256, 256)
                    self.layer_4 = nn.Linear(256, num_classes)
                    #obviously define your architecture here, layers, number of parameters etc.
            
                    optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
                    self.optimizer = optimizers[optimizer]
                    self.criterion = nn.CrossEntropyLoss()
                    self.learning_rate = learning_rate
                    self.num_classes = num_classes
            
                def forward(self, x):
                    batch_size, channels, height, width = x.size()
            
                    # define your forward function here
                    return x
            
                def configure_optimizers(self):
                    return self.optimizer(self.parameters(), lr=self.learning_rate)
            
                # logic for a single training step
                def training_step(self, batch, batch_idx):
                    # define your training step here
            
                    return {'loss': loss, 'accuracy': acc}
            
                # logic for a single validation step
                def validation_step(self, batch, batch_idx):
                    # define your validation step here
    
                    return {'loss': loss, 'accuracy': acc}
            
                # logic for a single testing step
                def test_step(self, batch, batch_idx):
                    # define your test step here
                    return {'loss': loss, 'accuracy': acc}
        ```
      4. **Step 4** After that you simply need to include two statements on tattooed.py file (one on the model switch IF and one on the dataset switch) and you are good to go. This can be done as follows:
      ```python
      # NOTE: Import your model and dataset class in tattooed.py
           ...
            elif cfg.dataset.name == 'YOUR DATASET NAME':
                data = DATASET_CLASS_NAME(base_path=Path(hydra.utils.get_original_cwd()),
                               batch_size=cfg.dataset.batch_size,
                               num_workers=cfg.dataset.num_workers)
        
           ...
            elif cfg.model.name == 'YOUR MODEL NAME':
                model = ResNet18(
                    num_classes=cfg.dataset.n_classes if not cfg.trainer.fine_tuning else old_dataset_num_classes,
                    learning_rate=cfg.model.lr)
        ```
    After doing so, you have, in three easy steps, that consist of mostly copying and modifying code from other files you have already pre-defined you are able to run TATTOOED on your new dataset and model architecture.



### License

`tattooed` was made with ♥ and it is released under the MIT license.