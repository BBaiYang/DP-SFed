from strategy.FedAVG import FedAVGTrainer
from strategy.DP_SFed import DPSFedTrainer
from configs import HYPER_PARAMETERS as hp

trainer = None
training_strategy = hp['training_strategy']
if training_strategy == 'DP_SFed':
    trainer = DPSFedTrainer()
elif training_strategy == 'FedAVG':
    trainer = FedAVGTrainer()
trainer.begin_train()
