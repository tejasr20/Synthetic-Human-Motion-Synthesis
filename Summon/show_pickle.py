import pickle

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print(type(data))
    print(data.keys())
    # return data

# Example usage:
pickle_file_path = '/data/tejasr20/summon/data/mdm/the_person_sits_down_on_a_wooden_chair/pickles/smplx.pickle'
loaded_data = load_pickle_file(pickle_file_path)
