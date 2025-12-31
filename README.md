jnp.zip includes pretrained models for all tasks.
It also includes task 1.1-1.3 resnet18 pretrained models (normal jnp_task1-1  to 1-3 models are efficientnet)
Rename to use resnet ones.


For teacher to run onsite test for pretrained models (included in jnp.zip):
-You can use TASK, BACKBONE and RUN:
    This runs for example model jnp_task1-3.pt located in the same directory.
    Example:    'python3 jnp.py TASK=1.3 RUN=1 BACKBONE=efficientnet'
    it skips training part and runs onsite test saving results to ./onsite_test_jnp_task1-3.csv

    Just change the test name to run all individually
    If it outputs error, you have chosen wrong backbone for this model (use BACKBONE=efficientnet or BACKBONE=resnet18)

    task 1.1 uses BACKBONE=efficientnet
    task 1.2 uses BACKBONE=efficientnet
    task 1.3 uses BACKBONE=efficientnet
    task 2.1 uses BACKBONE=resnet18
    task 2.2 uses BACKBONE=resnet18
    task 3.1 uses BACKBONE=efficientnet
    task 3.2 uses BACKBONE=efficientnet
    task 4 uses BACKBONE=efficientnet




How to generally run jnp.py:
-You can use TASK and BACKBONE:

    If you want to train and evaluate new model for specific task:
            Example:    'python4 jnp.py TASK=1.1 BACKBONE=efficientnet'

    If you want to only evaluate already trained model:
            Example:    'python4 jnp.py TASK=1.1 BACKBONE=efficientnet TRAIN=0'


-You can use manually all parameters (dont specify TASK)
    All available parameters: 'BACKBONE', 'TRAIN','EPOCHS', 'LEARNING_RATE' 'EVALUATE', 'FREEZE_BACKBONE', 'LOSS_FUNCTION', 'ATTENTION'
            Example:    'python4 jnp.py BACKBONE=resnet18 TRAIN=1, EPOCHS=20, EVALUATE=1'

