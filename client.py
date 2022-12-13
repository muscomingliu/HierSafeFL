# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
from models.initialize_model import initialize_model
import copy

class Client():
    def __init__(self, id, train_loader, test_loader, args, device, honest = True):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        #record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []
        self.honest = honest
        self.grad_history = 0
        self.device = device

    def local_update(self, num_iter, device):
        itered_num = 0
        loss = 0.0
        end = False
        self.grad_history = 0
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        iter = 1
        for epoch in range(iter):
            for data in self.train_loader:
                inputs, labels = data
                if not self.honest:
                    labels.apply_(lambda x: 7)
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels)
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            # layers = []
            # for name, layer in self.model.shared_layers.name_layer.items():
            #     if name == 'conv2_drop':
            #         continue
            #     layers.append(layer.weight.grad.flatten())
            #     layers.append(layer.bias.grad.flatten())
            # layers = torch.cat(layers).to(self.device)
            # if torch.linalg.norm(layers) > 1:
            #     layers = layers / torch.linalg.norm(layers)
            layers = self.model.shared_layers.fc2.weight.grad.flatten()
            layers = layers / torch.linalg.norm(layers)
            self.grad_history = torch.add(self.grad_history, layers)


            if end: break
            self.epoch += 1
            self.model.exp_lr_sheduler(epoch = self.epoch)
            # self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        loss /= num_iter
        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        message = {
            'client_id': self.id,
            'cshared_state_dict': copy.deepcopy(self.model.shared_layers.state_dict()),
            'grad_history': self.grad_history,
        }
        edgeserver.receive_from_client(message)
        return None

    def receive_from_edgeserver(self, message):
        self.receiver_buffer = message
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        self.model.update_model(self.receiver_buffer)
        return None

