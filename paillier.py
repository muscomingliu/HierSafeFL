from phe import paillier
def encrypt():
    public_key, private_key = paillier.generate_paillier_keypair()
    secret_number_list = [3.141592653, 300, -4.6e-12]
    encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]
    ret = [private_key.decrypt(x) for x in encrypted_number_list]
    print(ret)

def main():
    encrypt()
main()