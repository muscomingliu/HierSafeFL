# The structure of the server
# The server should include the following functions:
# 1. Server initialization
# 2. Server reveives updates from the user
# 3. Server send the aggregated information back to clients
import copy
import aggregator 
import torch

class Cloud():

    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        # self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.client_client_similarity = None

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict, client_reference_similarity):
        self.receiver_buffer[edge_id] = {'eshared_state_dict': eshared_state_dict,'client_reference_similarity':client_reference_similarity}
        return None

    def similarity_client_client(self, similarity_client_referecne):
        top_k = 10
        client_client_similarity = {}
        similarity_client_referecne = {k:v for tmp in similarity_client_referecne for k,v in tmp.items()}
        for client1 in similarity_client_referecne:
            for client2 in similarity_client_referecne:
                if client1 == client2 or (client2, client1) in client_client_similarity:
                    continue    
                client_client_similarity[(client1, client2)] = 0
                delta = torch.abs(torch.stack(similarity_client_referecne[client1]) - torch.stack(similarity_client_referecne[client2]))
                client_client_similarity[(client1, client2)] = torch.max(delta, client_client_similarity[(client1, client2)])
        return client_client_similarity

    def aggregate(self, args):
        eshared_state_dict = [dict['eshared_state_dict'] for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = aggregator.average_weights(w=eshared_state_dict,
                                                 s_num=sample_num)

        similarity_client_referecne = [dict['client_reference_similarity'] for dict in self.receiver_buffer.values()]
        self.client_client_similarity = self.similarity_client_client(similarity_client_referecne)
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict), self.client_client_similarity)
        return None

