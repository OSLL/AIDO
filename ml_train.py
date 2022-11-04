from ml import Trainer
from ml_model import ParallelLinearNet

if __name__ == "__main__":
    trainer = Trainer(ParallelLinearNet)
    trainer.train(100, 10, 10)