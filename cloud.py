# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated messagermation back to clients
import copy
import aggregator 
import torch
import random
from phe import paillier
import numpy as np
epsilon = 1e-5

class Cloud():

    def __init__(self, shared_layers, device):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.init_state = torch.flatten(shared_layers.fc2.weight)
        self.clock = []
        self.client_client_similarity = None
        self.reference_count = 50
        self.reference = self.get_reference()
        self.parameter_count = 0
        self.public_key, self.private_key = paillier.generate_paillier_keypair()
        self.s_prime = self.get_s_prime()
        self.client_reputation = None
        self.device = device
        self.client_learning_rate = None

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        edge.reference = self.reference.to(self.device)
        return None

    def receive_from_edge(self, message):
        edge_id = message['id']
        self.receiver_buffer[edge_id] = {'eshared_state_dict': message['eshared_state_dict'],
                                        'client_reference_similarity':message['client_reference_similarity']
                                    }
        return None

    def foolsgold(self, similarity_client_referecne):
        similarity_client_referecne = {k:v for tmp in similarity_client_referecne for k,v in tmp.items()}
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)
        
        n = len(similarity_client_referecne)
        cs = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cs[i][j] = cos(similarity_client_referecne[i], similarity_client_referecne[j]).item()
        #  Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99
        
        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0
        return cs, wv

    def contra(self, similarity_client_referecne):
        similarity_client_referecne = {k:v for tmp in similarity_client_referecne for k,v in tmp.items()}
        if self.client_reputation is None:
            self.client_reputation = np.ones((len(similarity_client_referecne)))
        if self.client_learning_rate is None:
            self.client_learning_rate = np.ones(len(similarity_client_referecne)) 

        n = len(similarity_client_referecne)
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-9)
        tao = np.zeros((n))
        topk = 5
        t = 0.5
        delta = 0.1

        cs = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cs[i][j] = cos(similarity_client_referecne[i], similarity_client_referecne[j]).item()

        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        for i in range(n):
            temp = np.argpartition(-cs[i], topk)
            tao[i] = np.mean(cs[i][temp[:topk]])
            if tao[i] > t:
                self.client_reputation[i] -= delta 
            else:
                self.client_reputation[i] += delta 

        #  Pardoning: reweight by the max value seen
        self.client_learning_rate = np.ones((n)) - tao
        self.client_learning_rate /= np.max(self.client_learning_rate)
        self.client_learning_rate = (np.log((self.client_learning_rate / (1 - self.client_learning_rate + epsilon)) + epsilon) + 0.5)
        self.client_learning_rate[(np.isinf(self.client_learning_rate) + self.client_learning_rate > 1)] = 1
        self.client_learning_rate[(self.client_learning_rate < 0)] = 0
        self.client_learning_rate /= np.sum(self.client_learning_rate)
        self.client_reputation /= max(self.client_reputation)
        return cs, self.client_reputation

    def aggregate(self, args):
        similarity_client_reference = [dict['client_reference_similarity'] for dict in self.receiver_buffer.values()]
        self.client_client_similarity, self.client_reputation = self.contra(similarity_client_reference)
   
        eshared_state_dict = [dict['eshared_state_dict'] for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = aggregator.average_weights_contra_cloud(w=eshared_state_dict, lr = self.client_learning_rate)
        return None

    def get_client_repuation(self, edge):
        client_repuation = {}
        for id in edge.cids:
            client_repuation[id] = self.client_reputation[id]
        return client_repuation

    def get_client_learning_rate(self, edge):
        client_learning_rate =  {}
        for id in edge.cids:
            client_learning_rate[id] = self.client_learning_rate[id]
        return client_learning_rate

    def send_to_edge(self, edge):
        client_reputation = self.get_client_repuation(edge)
        client_learning_rate = self.get_client_learning_rate(edge)
        message = {
            'shared_state_dict': self.shared_state_dict,
            'client_reputation': client_reputation,
            'client_learning_rate': client_learning_rate,
        }
        edge.receive_from_cloudserver(message)
        return None

    def get_reference(self):
        self.parameter_count = self.init_state.size()[0]
        self.parameter_count = int(self.parameter_count)

        nonzero_per_reference =  self.parameter_count // self.reference_count
        reference = torch.zeros((self.reference_count,  self.parameter_count))
        parameter_index_random = list(range( self.parameter_count))
        random.shuffle(parameter_index_random)

        for reference_index in range(self.reference_count):
            index = parameter_index_random[reference_index * nonzero_per_reference: (reference_index + 1) * nonzero_per_reference]
            index = torch.tensor(index)
            reference[reference_index][index] = 1
        # reference = torch.eye(self.parameter_count)
        return reference
    
    def get_s_prime(self):
        # a = random.sample(range(0, 100), self.reference.size()[0])
        # s = torch.matmul(self.reference.T, torch.tensor(a, dtype=torch.float32))
        # s_prime = [self.public_key.encrypt(x.item()) for x in s]
        return
        



