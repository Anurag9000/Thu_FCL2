# system/flcore/servers/serverGLFC.py
import copy
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientGLFC import clientGLFC
from flcore.servers.fcl_task_utils import prepare_task

class GLFCServer(Server):
    """
    GLFC server:
    - Keeps a pool of previous global models (proxy teacher); here we use the last best global as teacher.
    - Aggregates client prototypes (mean of normalized features) to broadcast if needed.
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        # Select slow clients if your base provides this API
        try:
            self.set_slow_clients()
        except Exception:
            pass

        # Create clients
        self.set_clients(clientGLFC)

        self.best_global = None
        self.global_prototypes = None  # [C, D]

    def train(self):
        for task in range(int(self.args.num_tasks)):
            prepare_task(self, task)
            self.current_task = task
            for rnd in range(self.global_rounds):
                glob_iter = task * self.global_rounds + rnd
                self._round_tag = glob_iter
                self.selected_clients = self.select_clients()
                teacher = self.best_global
                for client in self.selected_clients:
                    client.set_teacher(teacher)
                    self.send_models(client)
                    client.train()

                self.receive_models()
                self.aggregate_parameters()
                self.best_global = copy.deepcopy(self.global_model)
                self._aggregate_prototypes()
                self.test_metrics(task, glob_iter, flag="global")

        self.save_results()

    def _aggregate_prototypes(self):
        # Gather available client prototypes and average
        protos = []
        for c in self.selected_clients:
            p = getattr(c, "upload_prototypes", None)
            if p is not None:
                protos.append(p)
        if protos:
            P = torch.stack(protos, dim=0).float()
            self.global_prototypes = torch.nanmean(P, dim=0)  # [C, D]
