# Base Packages
import os
import wandb
import numpy as np
import pandas as pd
import argparse

# Tokenizattion
import spacy

# Datasets
from Preprocessing import AmazonReviewsDataset

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Progress Bar
from alive_progress import alive_bar, config_handler


class SelfAttention (nn.Module):
    """Self Attention implements the multi-headed dot product attention defined in the paper
    "Attention is all you need" . It has the capibility to be a masked attention layer if mask
    is not none in the forward function . 
    """

    def __init__(
        self,
        embedding_dim,
        heads
    ):
        """Initialize selfAttention .

        Args:
            embedding_dim (int): size of the embedding dimension
            heads (int) number of splits(heads) to use in multi headed attention
        """

        super(SelfAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.head_dim = embedding_dim//heads

        assert (self.head_dim * self.heads ==
                self.embedding_dim), 'embed size must be divisible by heads'

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # after concatenation of heads
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embedding_dim)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]

        # Lengths will mostly be sequence length for targets or src
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # split embedding into number of heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # query shape(n, query_len, heads, head_dim)
        # keys shape(n, key_len, heads, head_dim)
        # energy shape (n, heads, query_len, key_len)
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # Normalize across sentence(dim=3)
        attention = torch.softmax(energy/(self.embedding_dim**(1/2)), dim=3)

        # attention shape (N, heads, query_len, key_len)
        # after einsum (N, query_len, heads, head_dim)
        # out shape (N, query_len, embedding_dim or head_dim * heads)

        # print(
        #     f'attention: {attention.shape} values: {values.shape} value: {value_len} key: {key_len} query: {query_len} head_dim: {self.head_dim}')
        out = (
            torch.einsum(
                'nhql,nlhd->nqhd', [attention, values])
            .reshape(N, query_len, self.heads*self.head_dim)
        )

        out = self.fc_out(out)

        return out


class TransformerBlock (nn.Module):
    """Transformer block implements a single transformer layer 
    including a self attention block as its main driver with
    a feed forward and normalization layer following .
    Despite the use of a linear activiation, TransformerBlock cannot
    be used as a final output layer of a model; its output dimension is 
    not a percentage vector and instead (N, seq_len, embedding_dim) . 
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        forward_expansion=4,
        dropout=0.0
    ):
        """Initialize the TransformerBlock .

        Args:
            embedding_dim (int): Size of the embedding dimension
            num_heads (int): Number of heads to use in the self attention layer
            forward_expansion (int, optional): Multiple to expand dimensions by in the feed forward layer . Defaults to 4.
            dropout (float, optional): Dropout percentage Defaults to 0.0.
        """
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim*forward_expansion, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask):
        attended = self.attention(values, keys, queries, mask)

        # shouldnt queries+attention be larger than embedding_dim
        x = self.dropout(self.norm1(attended + queries))
        fed_forward = self.ff(x)
        out = self.dropout(self.norm2(fed_forward + x))  # too many dims?

        return out


class TransformerNet (nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        """TransformerNet is a classification network built on TranformerBlock layers .
        It creates positional embeddings then feeds inputs through TransformerBlock args.num_layers
        times.

        Args:
            args (argparse.Namespace) {
                    args.embedding_dim: size of embedding dimension
                    args.num_heads: number of heads to use in multi headed attention
                    args.forward_expansion: multiplier to expand dimensions by in TransformerBloc feed forward layers
                    args.dropout_rate: dropout percentage
                    args.num_layers: number of TransformerBlock layers
                }

            Kwargs {
                    device: torch.device to use
                    vocab_size: size of the input vocabulary
                    max_len: maximum sentence length
                }
        """

        super(TransformerNet, self).__init__()

        self.name = 'basic_transformer'

        self.embedding_dim = args.embedding_dim
        self.device = kwargs['device']

        self.word_embedding = nn.Embedding(
            kwargs['vocab_size'], self.embedding_dim)
        self.pos_embedding = nn.Embedding(
            kwargs['max_len'], self.embedding_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.embedding_dim,
                    args.num_heads,
                    args.forward_expansion,
                    args.dropout_rate
                ) for _ in range(args.num_layers)
            ]
        )

        self.dropout = nn.Dropout(args.dropout_rate)

        self.fc_out = nn.Linear(self.embedding_dim, 1)

    def forward(self, x, mask=None):
        N, seq_len = x.shape

        # Positional encoding
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(
            x) + self.pos_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        out = out.mean(dim=1)
        # reconsider taking only one dimension rather than using mean
        out = self.fc_out(out)

        return out.squeeze(dim=1)


class ConvolutionalSequenceNet (nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        # Add docstring and args layer num selection

        super(ConvolutionalSequenceNet, self).__init__()

        self.name = 'basic_cnn'

        filter_sizes = [1, 2, 3, 5]
        num_filters = 36

        self.embedding = nn.Embedding(
            kwargs['vocab_size'],
            embedding_dim=args.embedding_dim,
        )

        self.convs = nn.ModuleList([
            nn.Conv2d(
                1,
                num_filters,
                (filter_size, args.embedding_dim)
                # traverse through seq_len by filter size
                # this is done by stepping with embed size as the
                # second filter dim as to view the full word embed for each character
            ) for filter_size in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes)*num_filters, 1)

        self.layer_norm = nn.LayerNorm(num_filters)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        # shape: (N, C, seq_len, embed_size)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        normed = [self.layer_norm(layer) for layer in pooled]

        cat = self.dropout(torch.cat(normed, dim=1))

        out = self.fc(cat).squeeze(1)

        # DEBUG
        # print(*[output.shape for output in conved])
        # print(*[pool.shape for pool in pooled])
        # print(cat.shape)
        # print(out.shape)

        return out


class SentimentNet (nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        super(SentimentNet, self).__init__()

        self.name = 'basic_lstm'

        self.bidirectional = True if args.bidirectional == 't' else False

        self.output_size = kwargs['output_size']
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim

        self.embedding = nn.Embedding(
            kwargs['vocab_size'], embedding_dim=args.embedding_dim)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.sigmoid = nn.Sigmoid()

        self.fc = nn.Linear(
            args.hidden_dim * (int(self.bidirectional)+1), self.output_size)
        self.lstm = nn.LSTM(
            args.embedding_dim,
            args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout_rate,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, x):
        x = x.long()  # Assures propper datatyping
        batch_size = x.shape[0]

        embedded = self.embedding(x)

        # Lstm layer outputs too many dims for a linear layer so N and seq_len are temporarily flattened
        # together but are unfolded in the last out.view
        lstm_out, _ = self.lstm(embedded)  # shape (N, seq_len, hidden_size)
        # shape (N* seq_len, hidden_size)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim *
                                              (bool(self.bidirectional)+1))

        # Turns hidden_size dimension into 1
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # out = self.sigmoid(out) # Commented while using BCEwithlogits

        # unfolds N and seq_len back into dimentions
        out = out.view(batch_size, -1)
        out = out[:, -1]  # Gets only predictions at the last time step

        return out


def makedir_if_needed(directory):
    """Ensure directory if it doesn t exist .

    Args:
        directory ([path]): [path to create dir at]
    """
    if not os.path.isdir(directory):
        os.mkdir(directory)


def csv_scrape(directory):
    """Scrape all CSV files in a directory.

    Args:
        directory (str): String path of the directory to search for csv files

    Returns:
        [list]: List of str paths to csv files
    """
    assert os.path.isdir(directory)

    csv_files = []

    for cur_path, directories, files in os.walk(directory):
        for file in files:
            if os.path.splitext(
                    os.path.join(directory, cur_path, file))[1] == '.csv':

                # print (os.path.join(directory, cur_path, file))
                csv_files.append(str(os.path.join(directory, cur_path, file)))

    return csv_files


def check_for_improvement(epoch_accuracies, patience):
    if any(epoch_accuracies[-1] > epoch_loss for epoch_loss in epoch_accuracies[-patience:]):
        return True
    return False


def parse_args(running_dir):
    parser = argparse.ArgumentParser(description='Hyperparamaters')

    # Data Loading
    parser.add_argument('-dirname', '-d_name',
                        type=str,
                        help='Directory Name for csv data files',
                        required=False,
                        default='Reviews')

    parser.add_argument('-dataset_class', '-ds',
                        type=str,
                        help='Name of the class to load dataset with',
                        required=False,
                        default='AmazonReviews')

    parser.add_argument('-dataset_percent', '-dp',
                        type=float,
                        help='Percentage of the datast to use',
                        required=False,
                        default=0.03)

    parser.add_argument('-use_cuda', '-gpu',
                        type=str,
                        help='(t/f) Use cuda gpu for model training',
                        required=False,
                        default='t')

    parser.add_argument('-max_len', '-len',
                        type=int,
                        help='Maximum number of words in a single string example',
                        required=False,
                        default=80)

    parser.add_argument('-batch_size', '-bs',
                        type=int,
                        help='Batch size for dataloaders',
                        required=False,
                        default=128)

    parser.add_argument('-shuffle', '-s',
                        type=str,
                        help='(t/f) Shuffle data in trainloader',
                        required=False,
                        default='f')

    # Model Hyperparamaters
    parser.add_argument('-model', '-m',
                        type=str,
                        help='Name of the model to train',
                        required=False,
                        default='basic_lstm')

    parser.add_argument('-optimizer', '-optim',
                        type=str,
                        help='Name of the optimizer to use',
                        required=False,
                        default='Adam')

    parser.add_argument('-end_run_patience', '-early_stop_p',
                        type=int,
                        help='number of epochs to wait with no improvement before ending program',
                        required=False,
                        default=0)

    parser.add_argument('-decay_lr_patience', '-lr_p',
                        type=int,
                        help='number of epochs to wait with no improvement before decaying the learning rate',
                        required=False,
                        default=0)

    parser.add_argument('-embedding_dim', '-e_dim',
                        type=int,
                        help='Size of the model\'s embedding layer',
                        required=False,
                        default=300)

    parser.add_argument('-hidden_dim', '-h_dim',
                        type=int,
                        help='Size of the model\'s hidden layer',
                        required=False,
                        default=1024)

    parser.add_argument('-dropout_rate', '-dropout',
                        type=int,
                        help='Dropout rate during training cycle',
                        required=False,
                        default=0.4)

    parser.add_argument('-num_layers', '-n_layers',
                        type=int,
                        help='Number of layers to use for main block in the model',
                        required=False,
                        default=4)

    parser.add_argument('-num_heads', '-n_heads',
                        type=int,
                        help='Number of heads to use in multi headed attention blocks',
                        required=False,
                        default=8)

    parser.add_argument('-forward_expansion', '-expansion',
                        type=int,
                        help='Amount to temporarily expand dims during sequential linear blocks',
                        required=False,
                        default=4)

    parser.add_argument('-bidirectional', '-lstm_bi',
                        type=str,
                        help='(t/f)Transforms some archatectures to utilize bidirectionaly layers',
                        required=False,
                        default='f')

    # Training Hyperparamaters
    parser.add_argument('-learning_rate', '-lr',
                        type=float,
                        help='Learning rate for model training',
                        required=False,
                        default=0.006)

    parser.add_argument('-num_epochs', '-n_epochs',
                        type=int,
                        help='Number of epochs to train on',
                        required=False,
                        default=25)

    parser.add_argument('-max_norm', '-norm',
                        type=int,
                        help='maximium gradient clipping',
                        required=False,
                        default=5)

    parser.add_argument('-save_every', '-s_every',
                        type=int,
                        help='Save model each _ epochs',
                        required=False,
                        default=5)

    return parser.parse_args()


def load_data(args, running_dir):
    """Load data from a selected dataloading class.
    Dataloading classes load and preprocess data for the model

    Args:
        args (parser.arguments): {
            args.dataset_class(str): name of the dataloading class
            args.dirname (str): name of the directory to load data from
            args.max_len (int): maxium length for a src string
            args.batch_size(int): training and testing batch size
            args.shuffle (bool): whether to shuffle data in training
        }
        running_dir (os.path/str): directory the current python file is running in

    Raises:
        ValueError: When no csv files are found in the selected data dir

    Returns:
        # torch.utils.data.DataLoader abbreviated to Dataloader
        (Dataloader): trainloader with selected paramaters from args
        (Dataloader): tesloader with selected paramaters from args
        (dataset_class): initialized instance of the selected dataset class
        (dataset_class): test instance of selected dataset class
    """
    available_dataset_classes = {
        'AmazonReviews': AmazonReviewsDataset,
    }

    assert args.dataset_class in available_dataset_classes.keys(
    ), 'Selected DatasetClass not available'
    dataset_class = available_dataset_classes[args.dataset_class]

    data_dir = os.path.join(*[running_dir, 'Datasets', args.dirname])
    csv_files = csv_scrape(data_dir)
    if not len(csv_files) >= 1:
        print(f'Not enough csv files detected in {args.dirname}; \n Full path\
            {data_dir}')
        raise ValueError

    df = dataset_class.initialize_data(
        csv_files, args.dataset_percent).reset_index(drop=True)

    # Consistently uses 80% train set size for convienence
    trainset = df.iloc[:int(len(df)*0.80)]
    testset = df.iloc[int(len(df)*0.80):]

    trainset = dataset_class(
        trainset,
        max_len=args.max_len
    )

    testset = dataset_class(
        testset,
        max_len=args.max_len,
        vocab=[trainset.stoi, trainset.itos]
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        pin_memory=True
    )

    return trainloader, testloader, trainset, testset


def main(args, running_dir):
    available_models = {
        'basic_lstm': SentimentNet,
        'basic_cnn': ConvolutionalSequenceNet,
        'basic_transformer': TransformerNet
    }

    if args.use_cuda == 't' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'[Info] Using Device {device}')

    wandb.init(project='amazonreviewsclassification',
               config=args)

    trainloader, testloader, trainset, testset = load_data(args, running_dir)
    print(
        f'[Info] Dataloaders created from {args.dirname} with {len(trainloader)} batches')

    assert args.model in available_models.keys(), 'Selected model is not available'
    model = (available_models[args.model](
        args,
        vocab_size=len(
            trainset.stoi),
        output_size=1,
        device=device,
        max_len=args.max_len
    ).to(device))
    print(f'[Info] Model {args.model} Initialized')
    print(model)

    wandb.watch(model)

    ## Constants ##
    availabe_optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'SGD_nesterov': optim.SGD
    }

    uninitialized_optim = availabe_optimizers.get(args.optimizer)
    if args.optimizer == 'Adam':
        optimizer = uninitialized_optim(model.parameters(),
                                        lr=args.learning_rate)

    elif args.optimizer == 'SGD':
        optimizer = uninitialized_optim(model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0)

    elif (args.optimizer == 'SGD_nesterov'):
        optimizer = uninitialized_optim(model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=0.5,
                                        nesterov=True)

    print(
        f'[Info] Using Optimizer : {args.optimizer} with Plateau Decay {"on" if args.decay_lr_patience > 0 else "off"}')

    if args.decay_lr_patience > 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=args.decay_lr_patience
        )

    # BCE with logits automatically factors in a sigmoid and
    # allows for extra numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Get directory for all saved models
    saved_models_dir = os.path.join(running_dir, 'saved_models')
    makedir_if_needed(saved_models_dir)

    # Get directory for this specific run
    save_dir = os.path.join(
        saved_models_dir, f'archatecture{model.name}lr{args.learning_rate}batch_size{args.batch_size}\
num_layers{args.num_layers}embedding_size{args.embedding_dim}optim{args.optimizer}')

    makedir_if_needed(save_dir)
    print(f'[Info] Model Directory: {save_dir}')

    # Runs until another file of the same name at the same location isnt found
    # it increases the count each time allowing for another model
    count = 0
    while (count != -1):
        count += 1
        save_path = os.path.join(save_dir, f'model_{count}')
        if (os.path.isfile(save_path) != True):
            count = -1

    epoch_accuracies = list()
    print(f'Starting Training on {args.num_epochs} epochs')
    for epoch in range(1, args.num_epochs+1):
        model.train()
        epoch_losses = []
        with alive_bar(len(trainloader),
                       title='Training', bar='smooth',
                       length=75) as bar:
            for batch_num, batch in enumerate(trainloader):

                # Equivelant to model.zero_grad() but more efficient
                for param in model.parameters():
                    param.grad = None

                # Create src and trg batches
                src, trg = batch['src'].to(device), batch['trg'].to(device)

                output = model(src)

                loss = criterion(output, trg)
                epoch_losses.append(loss.item())
                loss.backward()

                # Gradient clipping to prevent vanishing/exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                optimizer.step()

                wandb.log(
                    {
                        'Loss': loss.item(),
                        'epoch': epoch
                    }
                )

                bar()
                bar.text(
                    f'Batch STD: {(torch.std (torch.sigmoid(output)) * 10**2).round() / (10**2)}')

        model.eval()

        test_losses = []
        test_accuracy = []
        with torch.no_grad():
            for batch in testloader:
                src, trg = batch['src'].to(device), batch['trg'].to(device)

                out = model(src)

                accuracy = (
                    torch.sum(
                        torch.round(
                            torch.sigmoid(out)
                        ) == trg)
                )/len(trg)

                loss = criterion(out, trg)

                test_accuracy.append(accuracy.item())
                test_losses.append(loss.item())

        if 'scheduler' in locals():
            scheduler.step(np.mean(test_losses))

        wandb.log(
            {
                'test loss': np.mean(test_losses),
                'accuracy': np.mean(test_accuracy),
                'epoch': epoch
            }
        )
        print(f'Epoch: {epoch}\tLoss: {np.round(np.mean(epoch_losses), decimals=3)}\
            \tTest Loss: {np.round(np.mean(test_losses), decimals=3)}\
            \tAccuracy: {np.mean(test_accuracy)}\
            \tLR: {scheduler.optimizer.param_groups[0]["lr"] if args.decay_lr_patience > 0 else args.learning_rate}\n'
              )

        if epoch % args.save_every == 0:
            checkpoint = {
                # saves all epochs in the same file with the epoch in their
                # indexable save name
                f'epoch:{epoch}_state_dict': model.state_dict(),
                f'epoch:{epoch}_optimizer': optimizer.state_dict()
            }

            # Save epoch to file
            torch.save(checkpoint, save_path)

        epoch_accuracies.append(np.mean(test_accuracy))
        if not check_for_improvement(epoch_accuracies, patience=args.end_run_patience) and args.end_run_patience > 0:
            break

    wandb.finish()

    print(
        "\033[0;32m" + "Program Finished Successfully" + "\033[0m")


if __name__ == '__main__':
    running_dir = os.path.dirname(os.path.realpath(__file__))

    args = parse_args(running_dir)
    main(args, running_dir)
