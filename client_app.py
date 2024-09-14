from models import UNetCompiled
from dataloader import create_splits,  load_custom_dataset, CustomDataset
from client import FlowerClient
input_dir = []
target_dir = [] 

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Loading the model and data 
    net = UNetCompiled().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    dataset = CustomDataset(input_dir, target_dir,tranform_img= True,transform_mask=True )
    splits, val_indices = create_splits(dataset,num_splits=5 )
    trainloader, valloader, _ = load_custom_dataset(partition_id,splits, val_indices,dataset)

    # Create a single Flower client representing a single organization
    # This creates an instance of the FlowerClient class that you defined
    return FlowerClient(net, trainloader, valloader, net, device=DEVICE).to_client()
