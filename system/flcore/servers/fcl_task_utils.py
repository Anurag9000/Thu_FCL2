"""Small shared task-boundary helper for continual-learning servers."""

from utils.data_utils import (
    read_client_data_FCL_cifar10,
    read_client_data_FCL_cifar100,
    read_client_data_FCL_imagenet1k,
)


def prepare_task(server, task):
    """Load the configured class-order task and refresh each client's labels."""
    if task == 0:
        all_labels = sorted({int(y) for c in server.clients for y in c.classes_so_far})
        current = sorted({int(y) for c in server.clients for y in c.current_labels})
        for client in server.clients:
            client.available_labels = list(all_labels)
            client.available_labels_current = list(current)
            client.available_labels_past = []
        return

    readers = {
        "CIFAR10": read_client_data_FCL_cifar10,
        "CIFAR100": read_client_data_FCL_cifar100,
        "IMAGENET1k": read_client_data_FCL_imagenet1k,
    }
    reader = readers.get(server.args.dataset)
    if reader is None:
        raise NotImplementedError(f"Task progression is unavailable for {server.args.dataset}")

    past = sorted({int(y) for y in server.clients[0].available_labels})
    for client in server.clients:
        data, info = reader(client.id, task=task, classes_per_task=server.args.cpt,
                            count_labels=True)
        client.next_task(data, info)

    all_labels = sorted({int(y) for c in server.clients for y in c.classes_so_far})
    current = sorted({int(y) for c in server.clients for y in c.current_labels})
    for client in server.clients:
        client.available_labels = list(all_labels)
        client.available_labels_current = list(current)
        client.available_labels_past = list(past)
