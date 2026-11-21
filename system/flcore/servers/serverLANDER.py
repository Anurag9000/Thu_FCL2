# system/flcore/servers/serverLANDER.py
import copy
import torch
from flcore.servers.serverbase import Server
from flcore.clients.clientLANDER import clientLANDER
from flcore.utils_core.text_encoder import get_text_anchors
from flcore.servers.fcl_task_utils import prepare_task

class LANDERServer(Server):
    """
    LANDER server:
    - Precompute/broadcast LTE anchors from class names (once).
    - Maintain previous global as teacher for optional KD synergy.
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        try:
            self.set_slow_clients()
        except Exception:
            pass
        self.set_clients(clientLANDER)

        self.best_global = None
        self._anchors = None

    def _ensure_anchors(self):
        if self._anchors is not None:
            return self._anchors
        # Get class names from any client's dataset if available
        class_names = None
        C = getattr(self.args, "num_classes", 100)
        class_names = [f"class_{i}" for i in range(C)]
        anchors = get_text_anchors(class_names, model_name=getattr(self.args, "lander_text_encoder", "clip-ViT-B-32"),
                                   device=self.device,
                                   template=getattr(self.args, "lander_text_template", "a photo of a {}"))
        self._anchors = anchors.cpu()
        return self._anchors

    def train(self):
        anchors = self._ensure_anchors()

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
                    client.set_anchors(anchors.to(self.device))
                    self.send_models(client)
                    client.train()

                self.receive_models()
                self.aggregate_parameters()
                self.best_global = copy.deepcopy(self.global_model)
                self.print_result(glob_iter)

        self.save_results()
