from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
from evaluator import Evaluator
from multiprocessing import Manager

if __name__ == '__main__':
    config = {
        'replay_buffer_size': 2048,
        'replay_buffer_episode': 400,
        'model_pool_size': 2,
        'model_pool_name': 'model-pool',
        'num_actors': 4,
        'episodes_per_actor': 8000,
        'gamma': 0.98,
        'lambda': 0.95,
        'min_sample': 200,
        'batch_size': 64,
        'epochs': 10,
        'clip': 0.2,
        'lr': 3e-4,
        'value_coeff': 1,
        'entropy_coeff': 0.01,
        'device': 'cuda',
        'ckpt_save_interval': 1800,
        'ckpt_save_path': 'LSTM_model/',
        'best_model_path': 'best_LSTM_model/',
        'eval_interval': 1,  # Sleep 1 seconds
        'eval_batch_size': 4,
        'mini_batch_size': 4
    }
    
    manager = Manager()
    #shared_list = manager.list()

    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    #evaluator_1 = Evaluator(config, shared_list)
    evaluator_1 = Evaluator(config)

    
    for actor in actors: actor.start()
    learner.start()
    evaluator_1.start()

    
    
    for actor in actors: actor.join()
    learner.terminate()
    evaluator_1.terminate()
    #print(list(shared_list))
