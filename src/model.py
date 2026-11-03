import os
import time
import torch
import argparse
from torch import device
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from src.utils import set_seed
from src.tokenizer import KmerTokenizer
from src.pairdataset import PairDataset

# def log1mexp(x: torch.Tensor):
#     """
#     Compute `log(1 - exp(-x))` in a numerically stable way for x > 0
#     """
#     log2 = torch.log(torch.tensor(2.0, dtype=x.dtype, device=x.device))
#     return torch.where(
#         x < log2, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))
#     )



class Model(torch.nn.Module):
    def __init__(self, input_dim:int, out_dim:int=100, seed:int=0):
        """
        Initialize the VIB model
        """
        super(Model, self).__init__()

        # Set the parameters
        self.__input_dim = input_dim
        self.__out_dim = out_dim
        self.__seed = seed

        # Set the seed
        set_seed(self.__seed)

        # Define the layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.__input_dim, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, self.__out_dim),
        )

    def forward(self, input: torch.Tensor):
        """
        Forward pass
        """
        embs = self.encoder(input)

        return embs

    def get_input_dim(self):
        """
        Return the input dimension
        """
        return self.__input_dim

    def get_out_dim(self):
        """
        Return the output dimension
        """
        return self.__out_dim

    def get_seed(self):
        """
        Return the seed value
        """
        return self.__seed

    def save(self, path:str):
        """
        Save the model
        """
        torch.save([
            {'input_dim': self.get_input_dim(), 'out_dim': self.get_out_dim(), 'seed': self.get_seed()},
            self.state_dict()
        ], path)


def loss_func(
        left_embs: torch.Tensor, right_embs: torch.Tensor, labels: torch.Tensor, loss_name:str = "mahalanobis",
):
    """
    Definition of the loss functions
    """

    if loss_name == "l2":

        squared_dist = torch.norm(left_embs - right_embs, dim=1) ** 2
        output = labels * squared_dist - (1 - labels) * torch.log(1 - torch.exp(-squared_dist) + 1e-6)
        return output.mean()

    elif loss_name == "bern_hinge_squared":

        # This loss is the function used in SemiBin2 implementation
        dist = torch.norm(left_embs - right_embs, p=2, dim=1)
        squared_dist = torch.square(dist)
        squared_margin = torch.square(torch.nn.functional.relu(1 - dist))
        output = labels * squared_dist + (1 - labels) * squared_margin

        return output.mean()

    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")


def train_batch(model, criterion, optimizer, left_kmers: torch.Tensor, right_kmers: torch.Tensor, labels: torch.Tensor):
    # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
    optimizer.zero_grad()

    # Get the embeddings
    left_embs = model.forward(left_kmers)
    right_embs = model.forward(right_kmers)

    # Compute the loss and backpropagate
    batch_loss = criterion(left_embs, right_embs, labels)
    batch_loss.backward()

    # Update the model parameters
    optimizer.step()

    return batch_loss

def train_single_epoch(device, model, criterion, optimizer, data_loader):

    epoch_loss = 0.
    for data in data_loader:

        left_kmers, right_kmers, labels = data
        left_kmers = left_kmers.reshape(-1, left_kmers.shape[-1]).to(device)
        right_kmers = right_kmers.reshape(-1, right_kmers.shape[-1]).to(device)
        labels = labels.reshape(-1).to(device)

        # Run the training for the current batch
        batch_loss = train_batch(
            model=model, criterion=criterion, optimizer=optimizer,
            left_kmers=left_kmers, right_kmers=right_kmers, labels=labels
        )

        # Get the epoch loss for reporting
        epoch_loss += batch_loss

    # Get the average epoch loss
    average_epoch_loss = epoch_loss.item() / len(data_loader)

    return average_epoch_loss

def train(
        device, distributed, model, lr, loss_name, loss_threshold, data_loader, epoch_num: int, save_every:int,
        output_path, summary_writer
):

    # Get the model parameters to train
    model_params = model.parameters()

    # Define the loss function.
    criterion = partial(loss_func, loss_name=loss_name)
    # Define the optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)

    ### Start training
    print(f"+ Training started.")

    # Set the correct training modes
    model.encoder.train()

    for current_epoch in range(epoch_num):

        # Set the epoch for the sampler
        if distributed:
            data_loader.sampler.set_epoch(current_epoch)

        # Run the training for the current epoch
        init_time = time.time()
        loss = train_single_epoch(
            device=device, model=model, criterion=criterion, optimizer=optimizer, data_loader=data_loader
        )
        loss_time = time.time() - init_time
        if distributed:
            if device == 0:
                if summary_writer is not None:
                    summary_writer.add_scalar('Loss/train', loss, current_epoch)
                    summary_writer.add_scalar('Time/epoch', loss_time, current_epoch)
                print(f"\t- Epoch: {current_epoch+1}/{epoch_num} - Loss: {loss} ({loss_time:.2f} secs)")
        else:
            if summary_writer is not None:
                summary_writer.add_scalar('Loss/train', loss, current_epoch)
                summary_writer.add_scalar('Time/epoch', loss_time, current_epoch)
            print(f"\t- Epoch: {current_epoch+1}/{epoch_num} - Loss: {loss} ({loss_time:.2f} secs)")

        ## Save the checkpoint if necessary
        if save_every > 0 and (current_epoch + 1) % save_every == 0:

            # Get the folder path of the output file and define the checkpoint path
            checkpoint_path = os.path.join(output_path+f".epoch_{current_epoch+1}.checkpoint")
            if distributed:
                if device == 0:
                    # Get the target folder path and if it does not exist, create it
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))

                    model.module.save(checkpoint_path)
            else:
                # Get the target folder path and if it does not exist, create it
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))

                model.save(checkpoint_path)

    if distributed:
        if device == 0:
            model.module.save(output_path)
            print(f"Model saved to {output_path}")
    else:
        model.save(output_path)
        print(f"+ Model saved to {output_path}")
    print(f"\t- Completed (Device {device}).")


def main_worker(
        device, world_size, distributed:bool, training_dataset: PairDataset, output_path: str, checkpoint: str,
        out_dim: int, lr:float, epoch_num:int, batch_size:int, loss_name: str, loss_threshold:float,
        workers_num: int, save_every: int, log_dir:str, seed: int,
):
    ### Initialize the device
    if distributed:
        # Set the environment variables for distributed training if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        print(f"+ Device: {device} | MASTER_ADDR: {os.environ['MASTER_ADDR']} | MASTER_PORT: {os.environ['MASTER_PORT']}")
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend='nccl', rank=device, world_size=world_size)

    # Define a DataLoader that iterates through batches of data.
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, num_workers=workers_num, pin_memory=True,
        sampler=DistributedSampler(training_dataset) if distributed else None,
        shuffle=False if distributed else True,
    )

    ### Define the model
    model = Model(input_dim=training_dataset.get_input_feature_dim(), out_dim=out_dim, seed=seed)
    if checkpoint != "":
        print(f"+ Loading the model from the checkpoint: {checkpoint}")
        # Load the model from the checkpoint
        kwargs, model_state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        model = Model(**kwargs)
        model.load_state_dict(model_state_dict)

    # Move the model to the device
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model.to(f"cuda:{device}"), device_ids=[device], find_unused_parameters=True)
    else:
        model.to(device)

    ### Define the log writer
    summary_writer = None
    if log_dir != "":
        summary_writer = SummaryWriter(log_dir=log_dir)

    ### Train the model
    train(
        device=device, distributed=distributed, model=model, lr=lr, loss_name=loss_name, loss_threshold=loss_threshold,
        data_loader=training_loader, epoch_num=epoch_num, save_every=save_every, output_path=output_path,
        summary_writer=summary_writer,
    )

    # Close the summary writer
    if summary_writer is not None:
        summary_writer.close()

    # Terminate the processes
    if distributed:
        # Wait for all processes to complete
        torch.distributed.barrier()
        # Release the distributed training resources
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument(
        '--input', type=str, help='Input DNA sequences file', required=True
    )
    parser.add_argument(
        '--input_type', type=str, help='Input file type (fasta, kmer)', required=False, default='fasta'
    )
    parser.add_argument(
        '--output', type=str, help='Model file path to save', required=True
    )
    parser.add_argument(
        '--checkpoint', type=str, default="", help='Checkpoint file path to load'
    )
    parser.add_argument(
        '--k', type=int, default=4, help='k value'
    )
    parser.add_argument(
        '--out_dim', type=int, default=256, help='Output dimension value'
    )
    parser.add_argument(
        '--neg_sample_per_pos', type=int, default=200, help='Negative samples count per positive'
    )
    parser.add_argument(
        '--max_seq_num', type=int, default=100000, help='Maximum number of sequences to get from the file'
    )
    parser.add_argument(
        '--epoch', type=int, default=50, help='Epoch number'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate'
    )
    parser.add_argument(
        '--loss_name', type=str, default="l2", help='Loss function (l2, bern_hinge_squared)'
    )
    parser.add_argument(
        '--loss_threshold', type=float, default=1.0, help='Loss threshold value'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100000, help='Batch size (0: no batch)'
    )
    parser.add_argument(
        '--save_every', type=int, default=5, help='Save checkpoint every x epochs (0 for no checkpoint)'
    )
    parser.add_argument(
        '--distributed', type=int, default=0, help='Use distributed training (0: no, 1: yes)'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', help='Device to use (cpu, mps or gpu)'
    )
    parser.add_argument(
        '--workers', type=int, default=2, help='Number of data loading workers'
    )
    parser.add_argument(
        '--log_dir', type=str, default="", help='Path directory to save the log file'
    )
    parser.add_argument(
        '--seed', type=int, default=1, help='Seed for random number generator'
    )

    args = parser.parse_args()

    # Define the world size, i.e. total number of processes participating in the distributed training job
    world_size = None
    if args.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        nodes_num = 1
        world_size = torch.cuda.device_count() * nodes_num
    else:
        if args.device == "gpu":
            assert torch.cuda.is_available(), "GPU is not available"
            device = torch.device(f"cuda")
        elif args.device == "mps":
            assert torch.backends.mps.is_available(), "MPS is not available"
            device = torch.device("mps")
        else:
            device = torch.device("cpu")


    print(f"+ Information")
    # Get the model name from the output path
    print(f"\t- k value: {args.k}")
    print(f"\t- Loss name: {args.loss_name}")
    print(f"\t- Loss threshold value: {args.loss_threshold}")
    print(f"\t- Epoch number: {args.epoch}")
    print(f"\t- Batch size: {args.batch_size}")
    print(f"\t- Learning rate: {args.lr}")
    print(f"\t- Output dimension: {args.out_dim}")
    print(f"\t- Negative sample per positive pair: {args.neg_sample_per_pos}")
    print(f"\t- Maximum number of sequence pairs to use: {args.max_seq_num}")
    if args.checkpoint != "":
        print(f"\t- Checkpoint file: {args.checkpoint}")
    if args.distributed:
        print(f"\t- Distributed training is activated with {world_size} GPUs")
    else:
        print(f"\t- No distributed training")
        print(f"\t- Device: {device}")
    print(f"\t- Number of data loading workers: {args.workers}")
    print(f"\t- Seed value: {args.seed}")
    print(f"\t- Output model name: {os.path.basename(args.output)}")

    ### Read the dataset and construct the tokenizer
    tokenizer = KmerTokenizer(k=args.k, counting_method = "canonical")
    training_dataset = PairDataset(
        file_path=args.input, tokenizer=tokenizer, neg_sample_per_pos=args.neg_sample_per_pos,
        input_type=args.input_type, max_seq_num=args.max_seq_num, seed=args.seed,
    )

    # Define the arguments for the main worker function
    arguments = (
        world_size, args.distributed, training_dataset, args.output, args.checkpoint, args.out_dim, args.lr,
        args.epoch, args.batch_size, args.loss_name, args.loss_threshold, args.workers, args.save_every,
        args.log_dir, args.seed,
    )
    if args.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=world_size, join=True, args=arguments)
    else:
        main_worker(*((device,) + arguments))
    print("+ Completed.")
