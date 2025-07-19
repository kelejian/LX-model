# PyTorch Template Project
# PyTorch 项目模板使用指南

本项目是一个基于配置驱动的 PyTorch 框架，旨在为深度学习项目提供一个标准化的、可复用的结构。

## 核心设计

该模板基于两大核心设计原则：

* **配置驱动 (Configuration-Driven)**: 项目的所有可变参数（如学习率、模型类型、数据集路径等）均由 `config.json` 文件集中管理。修改实验配置无需改动代码，便于参数调试和实验复现。
* **关注点分离 (Separation of Concerns)**: 项目代码按功能被严格划分到不同模块。每个模块职责单一（例如 `data_loader` 负责数据加载，`model` 负责模型定义），使代码结构清晰、易于维护和扩展。

## 目录结构

```
pytorch-template/
│
├── train.py                # 训练入口脚本
├── test.py                 # 测试/评估入口脚本
│
├── config.json             # 核心配置文件
├── parse_config.py         # 配置文件解析模块
│
├── base/                     # 抽象基类目录 (定义项目骨架)
│   ├── base_data_loader.py   # 数据加载器基类
│   ├── base_model.py         # 模型基类
│   └── base_trainer.py       # 训练器基类
│
├── data_loader/              # 数据加载实现
│
├── model/                    # 模型、损失函数、评估指标
│   ├── model.py
│   ├── loss.py
│   └── metric.py
│
├── trainer/                  # 训练流程实现
│   └── trainer.py
│
├── logger/                   # 日志与可视化模块
│
├── saved/                    # 默认输出目录 (保存模型和日志)
│
└── utils/                    # 通用工具函数模块
```

## 标准使用流程

1.  **准备组件**: 根据你的任务，在 `data_loader`, `model` 等目录中创建具体的实现类。
2.  **编写配置**: 创建一个 `config.json` 文件，在其中定义本次实验的名称、要使用的模型/数据加载器类型、超参数等。
3.  **启动训练**: 执行命令 `python train.py -c your_config.json`。
4.  **后台流程**:
    * `train.py` 调用 `ConfigParser` 解析配置，并自动创建本次实验的保存目录。
    * 根据配置，自动实例化数据加载器、模型、优化器等所有组件。
    * 所有组件被送入 `Trainer` 实例。
    * `Trainer` 开始执行训练循环，并由 `BaseTrainer` 负责处理日志记录、性能监控和检查点保存。
5.  **监控与分析**: 运行 `tensorboard --logdir saved/log` 启动 TensorBoard 服务，监控训练过程。
6.  **测试模型**: 训练完成后，使用 `test.py` 脚本加载指定的检查点（checkpoint）进行评估。

## 自定义任务指南

将此模板应用于新任务（如回归、生成模型等）的步骤如下：

### 第1步: 定义数据加载器 (`data_loader/`)

1.  在 `data_loader/data_loaders.py` 中创建一个新的类，继承自 `base.BaseDataLoader`。
2.  在 `__init__` 方法中，实现加载你自定义数据集的逻辑（例如，从CSV文件或图像文件夹加载）。
3.  确保你的数据集类返回的数据格式符合任务要求（例如，回归任务返回 `(特征, 浮点数目标值)`）。
4.  最后调用 `super().__init__(self.dataset, ...)` 将数据集和参数交给基类处理。基类会自动处理验证集划分。

### 第2步: 定义模型架构 (`model/model.py`)

1.  创建一个新的模型类，继承自 `base.BaseModel`。
2.  在 `__init__` 方法中，定义你模型所需的网络层。
3.  在 `forward` 方法中，实现数据的前向传播逻辑。
4.  **注意**: 对于回归任务，模型的最后一层不应有 `softmax` 或 `log_softmax` 等激活函数。

### 第3步: 定义损失与指标 (`model/loss.py`, `model/metric.py`)

1.  在 `model/loss.py` 中，添加适用于你任务的损失函数（如回归任务的 `mse_loss`）。
2.  在 `model/metric.py` 中，添加适用于你任务的评估指标（如回归任务的 `mae`）。分类任务的 `accuracy` 等指标应被移除或替换。

### 第4步: 修改配置文件 (`config.json`)

这是将新组件组装起来的关键一步。

1.  `name`: 为你的新实验指定一个清晰的名称。
2.  `arch.type`: 修改为你在第2步中定义的模型类名。
3.  `data_loader.type`: 修改为你在第1步中定义的数据加载器类名。
4.  `loss`: 修改为你在第3步中定义的损失函数名。
5.  `metrics`: 列表内容修改为你在第3步中定义的评估指标名。
6.  `monitor`: 设置用于监控模型性能的指标，例如 `"min val_mae"` 表示监控验证集上的平均绝对误差，并保存该值最小时的模型。

### 第5步: (高级) 修改训练逻辑 (`trainer/trainer.py`)

对于标准的监督学习任务（分类、回归），通常**无需**修改 `trainer/trainer.py`。

当任务的训练范式特殊时（如GAN的交替训练），你需要重写 `_train_epoch` 方法：

1.  在 `train.py` 中为不同的模型（如生成器G、判别器D）创建多个优化器。
2.  修改 `trainer/trainer.py` 的 `__init__` 方法，使其能接收多个模型和优化器。
3.  完全重写 `_train_epoch` 方法的内部逻辑，以实现你任务所需的特定训练步骤。`BaseTrainer` 提供的检查点管理等功能依然可用。

## 关键脚本命令

* **开始新训练**:
    ```bash
    python train.py -c config.json
    ```

* **从检查点恢复训练**:
    ```bash
    python train.py -r path/to/your/checkpoint.pth
    ```
    *注意: 此命令会自动加载同目录下的 `config.json`。如需使用新配置进行微调，可同时提供 `-c` 和 `-r` 参数。*

* **测试模型**:
    ```bash
    python test.py -r path/to/your/model_best.pth
    ```
    *该命令会加载指定检查点及其对应的 `config.json`，在测试集上进行评估。*

* **通过命令行修改配置**:
    *模板支持通过命令行临时覆盖 `config.json` 中的部分参数。*
    ```bash
    # 将学习率改为 0.0001，批大小改为 256
    python train.py -c config.json --lr 0.0001 --bs 256
    ```
    *可修改的快捷选项在 `train.py` 的 `options` 列表中定义。*
____________________________________________________________________________________________

* [PyTorch Template Project](#pytorch-template-project)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Config file format](#config-file-format)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
	* [Customization](#customization)
		* [Custom CLI options](#custom-cli-options)
		* [Data Loader](#data-loader)
		* [Trainer](#trainer)
		* [Model](#model)
		* [Loss](#loss)
		* [metrics](#metrics)
		* [Additional logging](#additional-logging)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
	* [Contribution](#contribution)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10	                 // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization
Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```
`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target` 
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss
Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization
This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server** 

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules. 

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

- [ ] Multiple optimizers
- [ ] Support more tensorboard functions
- [x] Using fixed random seed
- [x] Support pytorch native tensorboard
- [x] `tensorboardX` logger support
- [x] Configurable logging layout, checkpoint naming
- [x] Iteration-based training (instead of epoch-based)
- [x] Adding command line option for fine-tuning

