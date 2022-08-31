#from learning import Dataset
from learning import Trainer


if __name__ == "__main__":
    trainer = Trainer('model_with_real', 128, 2)
    trainer.train(300000)
    #trainer.evaluate('epoch_epoch_650')
