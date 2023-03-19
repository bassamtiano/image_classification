from utils.multiclass_preprocessor import MultiClassPreprocessor
from models.image_classifier import ImageClassifier

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    
    dm = MultiClassPreprocessor(batch_size = 50)
    
    model = ImageClassifier(lr = 1e-3, num_classes = 10)
    
    logger = TensorBoardLogger("logs", name = "cnn-cifar")
    
    trainer = pl.Trainer(
        accelerator = "gpu",
        max_epochs = 10,
        
        default_root_dir = "checkpoints/cnn_cifar",
        logger = logger
    )

    trainer.fit(model, datamodule = dm)
    output = trainer.test(model = model, datamodule = dm)