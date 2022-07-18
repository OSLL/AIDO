#from learning import Dataset
from learning import ParamsNet
from learning import Trainer


if __name__ == "__main__":
    trainer = Trainer('model', 32)
    trainer.train(1000000)
    #trainer.evaluate('epoch_epoch_650')
