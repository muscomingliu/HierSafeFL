# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated messagermation back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated messagermation from the cloud server

import copy
import aggregator 
import collections
import torch
import math 
from sklearn.metrics.pairwise import cosine_similarity
class Edge():

    def __init__(self, id, cids, shared_layers, device):
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
        # self.num_of_record = math.inf
        self.reference = None
        self.client_reference_similarity = {}
        self.history = {}
        self.device = device

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        self.id_registration.clear()
        # self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration[client.id] = {''}
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, message):
        client_id, cshared_state_dict, grad_history = message['client_id'], message['cshared_state_dict'], message['grad_history']
        if client_id not in self.history:
            self.history[client_id] = {'grad_history': 0, 'cshared_state_dict': None, 'reputation': 1, 'learning_rate': 1/20}
        self.history[client_id]['cshared_state_dict'] = cshared_state_dict
        self.history[client_id]['grad_history'] = torch.add(grad_history, self.history[client_id]['grad_history'])
        # self.history[client_id]['grad_history'] = grad_history
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def _average_record(self, client_id):
        # receiver_buffer = self.receiver_buffer[client_id]
        # w_avg = []

        # for i in range(len(receiver_buffer)):   
        #     tmp = ([torch.flatten(receiver_buffer[i][k]) for k in receiver_buffer[0].keys()])
        #     w_avg.append(torch.cat(tmp))
        # w_avg = torch.mean(torch.stack(w_avg), axis = 0)

        # return w_avg````
        return self.history[client_id]['grad_history']
        
    def _similarity(self, average_record):
        ret = torch.matmul(self.reference, average_record)
        # ret = average_record
        return ret


    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        # sample_num = {k: v for k, v in self.sample_registration.items()}
        # self.shared_state_dict = aggregator.average_weights_edge(history = self.history,
        #                                          s_num= sample_num)

        # received_dict = [dict for dict in self.receiver_buffer.values()]
        # sample_num = [snum for snum in self.sample_registration.values()]
        # self.shared_state_dict =  aggregator.average_weights(w = received_dict,
        #                                          s_num= sample_num)

        self.shared_state_dict = aggregator.average_weights_contra(history = self.history)

      

    def send_to_client(self, client):
        message = {'shared_state_dict':copy.deepcopy(self.shared_state_dict),}
        client.receive_from_edgeserver(message)
        return None

    def send_to_cloudserver(self, cloud):
        for client_id in self.history:
            average_record = self._average_record(client_id)
            self.client_reference_similarity[client_id] = self._similarity(average_record)
        message =  {'eshared_state_dict': copy.deepcopy(self.shared_state_dict),
                'client_reference_similarity': copy.deepcopy(self.client_reference_similarity),
                'id':self.id,
        }
        cloud.receive_from_edge(message)
        return None

    def receive_from_cloudserver(self, message):
        shared_state_dict, client_reputation, client_learning_rate = message['shared_state_dict'], message['client_reputation'], message['client_learning_rate']
        self.shared_state_dict = shared_state_dict
        for id, reputation in client_reputation.items():
            self.history[id]['reputation'] = reputation
        for id, client_learning_rate in client_learning_rate.items():
            self.history[id]['learning_rate'] = client_learning_rate
        return None

   