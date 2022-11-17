# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
import aggregator 
import collections
import torch

class Edge():

    def __init__(self, id, cids, shared_layers):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = {}
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.num_of_record = 10
        self.reference = []
        self.client_reference_similarity = {}

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.id_registration.clear()
        self.sample_registration.clear()
        self.client_reference_similarity.clear()
        return None

    def client_register(self, client):
        self.id_registration[client.id] = {'repuatation': 1}
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict):
        if client_id not in self.receiver_buffer:
            self.receiver_buffer[client_id] = collections.deque()
        if len(self.receiver_buffer[client_id]) >= self.num_of_record:
            self.receiver_buffer[client_id].popleft()
        self.receiver_buffer[client_id].append(cshared_state_dict)
        return None

    def _average_record(self, client_id):
        receiver_buffer = self.receiver_buffer[client_id]
        w_avg = []

        for i in range(len(receiver_buffer)):   
            tmp = ([torch.flatten(receiver_buffer[i][k]) for k in receiver_buffer[0].keys()])
            w_avg.append(torch.cat(tmp))
        w_avg = torch.mean(torch.stack(w_avg), axis = 0)
        return w_avg
        
    def _similarity(self, average_record):
        # the fake referece 
        self.reference = [copy.deepcopy(average_record)] * 5
        similarity = [None] * len(self.reference)
        for i, reference in enumerate(self.reference):
            dot = average_record.dot(reference)
            norm = torch.norm(average_record) * torch.norm(reference)
            similarity[i] = dot / norm
        return similarity

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dq[-1] for dq in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = aggregator.average_weights(w = received_dict,
                                                 s_num= sample_num)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        for client_id in self.receiver_buffer:
            average_record = self._average_record(client_id)
            self.client_reference_similarity[client_id] = self._similarity(average_record)

        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict),
                                client_reference_similarity = copy.deepcopy(
                                    self.client_reference_similarity)
                                )
        return None

    def receive_from_cloudserver(self, shared_state_dict, client_client_similarity):
        self.shared_state_dict = shared_state_dict
        self.client_client_similarity = client_client_similarity

        return None

